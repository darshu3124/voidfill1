import cv2
import numpy as np
import os

def get_answer_box(image):
    # Resize for consistent processing (from temp/app.py)
    height, width = image.shape[:2]
    new_width = 800
    ratio = new_width / float(width)
    resized = cv2.resize(image, (new_width, int(height * ratio)))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 100:
            return resized[y:y+h, x:x+w]

    return resized

def capture_choices_hough(image):
    """Detect bubbles using HoughCircles logic from temp/app.py"""
    answer_box = get_answer_box(image)
    gray = cv2.cvtColor(answer_box, cv2.COLOR_BGR2GRAY)
    
    # Threshold for filling detection: white pixels are marks
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # DETECT CIRCLES (Reference Logic: temp/app.py)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=8,
        maxRadius=25
    )

    if circles is None: return {}, [], answer_box, thresh

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda c: c[1]) # Top-to-bottom

    rows = []
    if len(circles) > 0:
        current_row = [circles[0]]
        for c in circles[1:]:
            if abs(c[1] - current_row[0][1]) < 20: 
                current_row.append(c)
            else:
                rows.append(current_row)
                current_row = [c]
        rows.append(current_row)

    student_answers = {}
    row_data_for_marking = []
    q_no = 1

    for row in rows:
        row = sorted(row, key=lambda c: c[0]) # Left-to-right
        for i in range(0, len(row), 4):
            group = row[i:i+4]
            if len(group) < 4: continue

            pixel_vals = []
            for (x, y, r) in group:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mask, (x, y), int(r * 0.8), 255, -1)
                pixel_vals.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))

            max_val = max(pixel_vals)
            marked_cols = []
            
            if max_val < 50: 
                 sel_ans = "BLANK"
            else:
                marked_cols = [idx for idx, v in enumerate(pixel_vals) if v > 0.7 * max_val and v > 50]
                if len(marked_cols) == 1:
                    sel_ans = chr(65 + marked_cols[0]) # A, B, C, D
                elif len(marked_cols) > 1:
                    sel_ans = "INVALID"
                else:
                    sel_ans = "BLANK"

            student_answers[q_label := f"Q{q_no}"] = sel_ans
            row_data_for_marking.append((q_no, group, marked_cols if len(marked_cols) == 1 else [] if sel_ans == "BLANK" else marked_cols))
            q_no += 1

    return student_answers, row_data_for_marking, answer_box, thresh

def draw_captured_choices(warped, row_data):
    """Draw choices based on the Hough detected circles"""
    for (q_num, group, marked_cols) in row_data:
        for idx in marked_cols:
            (x, y, r) = group[idx]
            color = (255, 0, 0) # BLUE
            cv2.circle(warped, (x, y), r + 2, color, 3) # Circle outline
            cv2.circle(warped, (x, y), 5, color, -1) # Center dot
    return warped

def process_omr(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None: raise ValueError("Could not read image.")
    
    # STEP 1: CAPTURE CHOICES (using HoughCircles logic)
    selected, row_data, best_warped, _ = capture_choices_hough(image)
    
    # STEP 2: DRAW CAPTURED MARKINGS
    marked_image = draw_captured_choices(best_warped, row_data)
    cv2.imwrite(output_path, marked_image)
    
    return 0, len(selected), selected, output_path
