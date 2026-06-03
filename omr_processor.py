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
                                   cv2.THRESH_BINARY_INV, 51, 10)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        solidity = area / float(w * h) if w * h > 0 else 0
        
        # Corner marks are small black squares on the sheet
        if 50 < area < (width * height * 0.05):
            if 0.65 <= aspect_ratio <= 1.55 and solidity > 0.5:
                centers.append((x + w//2, y + h//2))
    
    # Group candidate centers into the 4 quadrants of the image
    quad_tl = [pt for pt in centers if pt[0] < width / 2 and pt[1] < height / 2]
    quad_tr = [pt for pt in centers if pt[0] >= width / 2 and pt[1] < height / 2]
    quad_bl = [pt for pt in centers if pt[0] < width / 2 and pt[1] >= height / 2]
    quad_br = [pt for pt in centers if pt[0] >= width / 2 and pt[1] >= height / 2]
    
    # We must have at least one corner mark candidate in each quadrant to perform a stable warp
    if not (quad_tl and quad_tr and quad_bl and quad_br):
        # Fallback: if we don't have all 4 quadrants represented, resize and proceed without warping
        return cv2.resize(image, (800, 1100))
    
    # Select the candidate in each quadrant closest to that respective corner of the page
    pt_tl = min(quad_tl, key=lambda pt: pt[0]**2 + pt[1]**2)
    pt_tr = min(quad_tr, key=lambda pt: (pt[0] - width)**2 + pt[1]**2)
    pt_bl = min(quad_bl, key=lambda pt: pt[0]**2 + (pt[1] - height)**2)
    pt_br = min(quad_br, key=lambda pt: (pt[0] - width)**2 + (pt[1] - height)**2)
    
    src_pts = np.float32([pt_tl, pt_tr, pt_bl, pt_br])
    
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
                                   cv2.THRESH_BINARY_INV, 31, 20)

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
    # Using 20px threshold to merge duplicate concentric contours of the same bubble.
    all_candidates = []
    for cand in temp_candidates:
        is_dup = False
        for final in all_candidates:
            dist = np.sqrt((cand['center'][0] - final['center'][0])**2 + (cand['center'][1] - final['center'][1])**2)
            if dist < 20: 
                is_dup = True
                # Keep the larger contour/bounding box to ensure we get the full bubble outline
                if cand['rect'][2] * cand['rect'][3] > final['rect'][2] * final['rect'][3]:
                    final['rect'] = cand['rect']
                    final['center'] = cand['center']
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
    bubble_metadata = {} # Stores coords per question
    shaded_metadata = {} # Stores which options were shaded per question
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
                        # Using 0.6 factor to target only the center of the bubble, ignoring the border.
                        cv2.circle(mask, b['center'], int(min(b['rect'][2], b['rect'][3]) // 2 * 0.6), 255, -1)
                        mask_pixels = cv2.countNonZero(mask)
                        filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                        pixel_vals.append((filled_pixels / mask_pixels) * 100)
                    
                    total_q_found += 1
                    q_label = f"Q{total_q_found}"
                    bubble_metadata[q_label] = [b['center'] for b in group]
                    
                    # --- FINAL ROBUST SELECTION LOGIC ---
                    max_idx = pixel_vals.index(max(pixel_vals))
                    sorted_vals = sorted(pixel_vals, reverse=True)
                    best_fill = max(pixel_vals)

                    # Track which bubbles in this group are actually shaded
                    # (Must be high absolute fill AND close to the best fill in this row)
                    shaded_this_q = []
                    if best_fill >= 35:
                        for idx, fill in enumerate(pixel_vals):
                            if fill >= 35 and (fill / best_fill) >= 0.75:
                                shaded_this_q.append(chr(65 + idx))
                    shaded_metadata[q_label] = shaded_this_q
                    
                    if best_fill < 35: # Must be at least 35% filled to count as any selection
                        student_answers[q_label] = "BLANK"
                    elif len(sorted_vals) > 1:
                        # Only mark INVALID if the second-best is high and close to the best one (double shading)
                        if sorted_vals[1] > 30 and (sorted_vals[1] / sorted_vals[0]) > 0.75:
                            student_answers[q_label] = "INVALID"
                        else:
                            student_answers[q_label] = chr(65 + max_idx)
                    else:
                        student_answers[q_label] = chr(65 + max_idx)
                else:
                    i += 1
            else:
                break

    # Store metadata for visual feedback
    student_answers["_bubble_metadata"] = bubble_metadata
    student_answers["_shaded_metadata"] = shaded_metadata
    return student_answers

def process_omr(image_path, output_path, answer_key=None):
    """Interface for app.py with strict, isolated visual feedback."""
    image = cv2.imread(image_path)
    if image is None: return 0, 0, {}, ""
    warped = get_perspective_transform(image)
    
    results = collect_selected_answers(image)
    if "error" in results:
        cv2.imwrite(output_path, warped)
        return 0, 0, {}, output_path

    # Extract bubble groups and remove metadata before returning to app.py
    bubble_metadata = results.pop("_bubble_metadata", {})
    shaded_metadata = results.pop("_shaded_metadata", {})
    
    qr_data = None
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(warped)
        if data: qr_data = data
    except: pass
    if qr_data: results["_qr_code"] = qr_data

    # Drawing feedback: Highlight based on correctness if answer_key is provided
    for q_label, coords in bubble_metadata.items():
        student_ans = results.get(q_label)
        correct_ans = answer_key.get(q_label) if answer_key else None
        shaded_opts = shaded_metadata.get(q_label, [])
        
        for i, (cx, cy) in enumerate(coords):
            opt = chr(65 + i)
            
            if not answer_key:
                # If no key, just show detected options in Blue
                cv2.circle(warped, (cx, cy), 15, (255, 0, 0), 2)
                continue

            # With Answer Key: ONLY draw relevant feedback
            is_shaded = opt in shaded_opts
            
            if is_shaded:
                # 1. If it's correct AND it's the ONLY thing shaded -> Green
                if opt == correct_ans and student_ans != "INVALID":
                    cv2.circle(warped, (cx, cy), 15, (0, 255, 0), 3)
                else:
                    # 2. If it's shaded but wrong OR part of INVALID (multiple) -> Red
                    cv2.circle(warped, (cx, cy), 15, (0, 0, 255), 3)
            elif opt == correct_ans:
                # 3. If it's the correct answer but the student missed it -> Blue
                cv2.circle(warped, (cx, cy), 15, (255, 0, 0), 3)
            # Else: Skip drawing for unshaded, incorrect bubbles to keep UI clean
    
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