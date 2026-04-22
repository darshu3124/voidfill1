import cv2
import numpy as np
import os
import json

# -----------------------------
# CORE OMR SCANNING LOGIC (ROBUST)
# -----------------------------

def get_perspective_transform(image):
    """Detects 4 corner marks and warps the image for alignment."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold to find the marks (black squares)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            
            # Looking for the corner squares (usually small relative to page but distinct)
            if area > 100 and 0.8 <= aspect_ratio <= 1.2:
                marks.append(approx)
    
    if len(marks) < 4:
        # Fallback: if we can't find 4 marks, resize and proceed without warp
        return cv2.resize(image, (800, 1100))

    # Get centroids
    centers = []
    for m in marks:
        M = cv2.moments(m)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    
    # Sort centers: top-left, top-right, bottom-left, bottom-right
    centers = sorted(centers, key=lambda x: x[1])
    top = sorted(centers[:2], key=lambda x: x[0])
    bottom = sorted(centers[2:], key=lambda x: x[0])
    
    src_pts = np.float32([top[0], top[1], bottom[0], bottom[1]])
    
    # Define destination points (A4ish aspect ratio)
    dest_w = 800
    dest_h = 1100
    dest_pts = np.float32([[20, 20], [dest_w-20, 20], [20, dest_h-20], [dest_w-20, dest_h-20]])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
    warped = cv2.warpPerspective(image, matrix, (dest_w, dest_h))
    
    return warped

def collect_selected_answers(image):
    """Refined logic to isolates ONLY the question section by bubble density."""
    warped = get_perspective_transform(image)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 31, 15)

    # 1. Find ALL circles on the page (must use LIST or TREE to find them inside boxes)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    temp_candidates = []
    for c in cnts:
        (bx, by, bw, bh) = cv2.boundingRect(c)
        # Filter for bubble size
        if 13 <= bw <= 55 and 13 <= bh <= 55 and 0.7 <= (bw/bh) <= 1.3:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = 4 * np.pi * (area / (peri * peri))
                # Circularity 0.78 to be safe but exclude very square boxes
                if circularity > 0.78:
                    temp_candidates.append({'center': (bx + bw//2, by + bh//2), 'rect': (bx, by, bw, bh)})

    # Robust duplicate removal (removing overlapping contours like letters vs bubbles)
    all_candidates = []
    for cand in temp_candidates:
        is_dup = False
        for final in all_candidates:
            dist = np.sqrt((cand['center'][0] - final['center'][0])**2 + (cand['center'][1] - final['center'][1])**2)
            if dist < 15: # Merge if centroids are very close
                # Keep the one with larger area (likely the bubble outline)
                is_dup = True
                break
        if not is_dup:
            all_candidates.append(cand)

    # 2. Identify the Question Boxes
    # We look for the box that contains the most candidates
    ext_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_roi = None
    max_count = 0
    
    for c in ext_cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 100 or h < 30: continue
        
        # Count bubbles in this external rect
        count = sum(1 for cand in all_candidates if x < cand['center'][0] < x+w and y < cand['center'][1] < y+h)
        if count > max_count:
            max_count = count
            best_roi = (x, y, w, h)

    # Filter candidates to only those in the best ROI (if found and significant)
    if best_roi and max_count >= 3:
        x, y, w, h = best_roi
        bubbles = [b for b in all_candidates if x < b['center'][0] < x+w and y < b['center'][1] < y+h]
    else:
        # If no clear box, fallback to candidates that are likely question options (top/bottom filtered)
        bubbles = [b for b in all_candidates if 150 < b['center'][1] < 1000]

    if not bubbles:
        return {"error": "No options detected. Please ensure the camera is close to the bubbles."}

    # 3. Group bubbles into rows
    bubbles.sort(key=lambda b: b['center'][1])
    rows = []
    if bubbles:
        curr_row = [bubbles[0]]
        for i in range(1, len(bubbles)):
            if abs(bubbles[i]['center'][1] - curr_row[-1]['center'][1]) < 25: 
                curr_row.append(bubbles[i])
            else:
                rows.append(sorted(curr_row, key=lambda b: b['center'][0]))
                curr_row = [bubbles[i]]
        rows.append(sorted(curr_row, key=lambda b: b['center'][0]))

    student_answers = {}
    validated_bubble_coords = []
    total_q_found = 0
    
    # 4. Group row items into clusters of 4
    for row in rows:
        i = 0
        while i < len(row):
            if i + 3 < len(row):
                group = row[i:i+4]
                gaps = [group[j+1]['center'][0] - group[j]['center'][0] for j in range(3)]
                
                # Broaden gap tolerance slightly for tilted scans
                if max(gaps) < 85:
                    i += 4
                    pixel_vals = []
                    for b in group:
                        mask = np.zeros(thresh.shape, dtype="uint8")
                        cv2.circle(mask, b['center'], int(min(b['rect'][2], b['rect'][3]) // 2 * 0.8), 255, -1)
                        mask_pixels = cv2.countNonZero(mask)
                        filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        pixel_vals.append((filled_pixels / mask_pixels) * 100)
                        validated_bubble_coords.append(b['center'])
                    
                    total_q_found += 1
                    q_label = f"Q{total_q_found}"
                    # --- FINAL ROBUST SELECTION LOGIC ---
                    max_idx = pixel_vals.index(max(pixel_vals))
                    sorted_vals = sorted(pixel_vals, reverse=True)
                    best_fill = max(pixel_vals)
                    
                    if best_fill < 45: # Must be at least 45% filled to count as any selection
                        student_answers[q_label] = "BLANK"
                    elif len(sorted_vals) > 1:
                        # Only mark INVALID if the second-best is VERY high (actual double shading)
                        # and very close to the best one.
                        if sorted_vals[1] > 40 and (sorted_vals[1] / sorted_vals[0]) > 0.85:
                            student_answers[q_label] = "INVALID"
                        else:
                            student_answers[q_label] = chr(65 + max_idx)
                    else:
                        student_answers[q_label] = chr(65 + max_idx)
                else:
                    i += 1
            else:
                break

    # Store validated coords for visual feedback
    student_answers["_validated_bubbles"] = validated_bubble_coords
    return student_answers

def process_omr(image_path, output_path):
    """Interface for app.py with strict, isolated visual feedback."""
    image = cv2.imread(image_path)
    if image is None: return 0, 0, {}, ""
    warped = get_perspective_transform(image)
    
    results = collect_selected_answers(image)
    if "error" in results:
        cv2.imwrite(output_path, warped)
        return 0, 0, {}, output_path

    # Extract validated bubbles and remove metadata before returning to app.py
    validated_bubbles = results.pop("_validated_bubbles", [])
    
    qr_data = None
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(warped)
        if data: qr_data = data
    except: pass
    if qr_data: results["_qr_code"] = qr_data

    # Drawing feedback: Highlight ONLY bubbles that were actually part of a validated question
    for (cx, cy) in validated_bubbles:
        cv2.circle(warped, (cx, cy), 15, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, warped)
    q_count = len([k for k in results if k.startswith("Q")])
    return 0, q_count, results, output_path

def extract_answers(image_path):
    """Interface for extracting answer keys from a filled sheet"""
    image = cv2.imread(image_path)
    if image is None:
        return {}

    student_answers = collect_selected_answers(image)
    if "error" in student_answers:
        return {}

    extracted = {}
    for k, v in student_answers.items():
        if k.startswith("Q"):
            try:
                q_num = int(k[1:])
                extracted[q_num] = v
            except: pass
    return extracted