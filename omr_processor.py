from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import io

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# CORE OMR SCANNING LOGIC
# -----------------------------
def get_answer_box(image):
    # Resize for consistent processing
    height, width = image.shape[:2]
    new_width = 800
    ratio = new_width / float(width)
    resized = cv2.resize(image, (new_width, int(height * ratio)))
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Use RETR_TREE to find hierarchical relationship (bubbles inside boxes)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return resized

    hierarchy = hierarchy[0]
    best_box = None
    max_bubbles = -1

    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        # Filter for potential question container boxes
        if w > 150 and h > 40:
            bubbles_in_box = 0
            child_idx = hierarchy[i][2]
            while child_idx != -1:
                child_cnt = cnts[child_idx]
                (iw, ih) = cv2.boundingRect(child_cnt)[2:]
                iar = iw / float(ih)
                area = cv2.contourArea(child_cnt)
                perimeter = cv2.arcLength(child_cnt, True)
                
                # Heuristics for a real bubble:
                # 1. Aspect ratio near 1.0
                # 2. Size between 10 and 60 pixels (at 800px width)
                # 3. Circularity check
                if 12 <= iw <= 60 and 12 <= ih <= 60 and 0.7 <= iar <= 1.3:
                    if perimeter > 0:
                        circularity = 4 * 3.14159 * (area / (perimeter * perimeter))
                        if circularity > 0.5: # Corrected check for filled/unfilled circles
                            bubbles_in_box += 1
                child_idx = hierarchy[child_idx][0]
            
            if bubbles_in_box > max_bubbles:
                max_bubbles = bubbles_in_box
                best_box = (x, y, w, h)

    if best_box and max_bubbles >= 4:
        x, y, w, h = best_box
        pad = 8
        y1, y2 = max(0, y-pad), min(resized.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(resized.shape[1], x+w+pad)
        return resized[y1:y2, x1:x2]

    # Fallback with sorted area but still requiring some width
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.4 * new_width and h > 40:
            return resized[y:y+h, x:x+w]

    return resized


def collect_selected_answers(image):
    """The Logic to collect choices directly from the scanned sheet"""
    answer_box = get_answer_box(image)
    gray = cv2.cvtColor(answer_box, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing: Use adaptive thresholding for better resilience to lighting/shadows
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Use RETR_LIST to find bubbles (more exhaustive than external only)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # Consistent filtering with detection phase but slightly more permissive for filled bubbles
        if 10 <= w <= 65 and 10 <= h <= 65 and 0.6 <= ar <= 1.4:
            if perimeter > 0:
                circularity = 4 * 3.14159 * (area / (perimeter * perimeter))
                if circularity > 0.45:
                    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                    bubbles.append((int(cX), int(cY), int(radius)))

    if len(bubbles) < 4:
        # If still failing, return detail for debugging
        return {"error": f"Only detected {len(bubbles)} potential bubbles inside the box."}
    
    # Remove duplicates (overlapping contours)
    bubbles = sorted(bubbles, key=lambda b: b[1])
    clean_bubbles = []
    for b in bubbles:
        is_dup = False
        for cb in clean_bubbles:
            dist = np.sqrt((b[0]-cb[0])**2 + (b[1]-cb[1])**2)
            if dist < 15: # Increased distance threshold
                is_dup = True
                break
        if not is_dup:
            clean_bubbles.append(b)
    bubbles = clean_bubbles

    # Sort bubbles by their Y-coordinate (top to bottom)
    bubbles = sorted(bubbles, key=lambda b: b[1])

    # Group bubbles into rows based on vertical proximity
    rows = []
    if bubbles:
        curr_row = [bubbles[0]]
        for b in bubbles[1:]:
            # Increased threshold for row grouping to be more tolerant of tilt
            if abs(b[1] - curr_row[0][1]) < 40: 
                curr_row.append(b)
            else:
                rows.append(sorted(curr_row, key=lambda x: x[0]))
                curr_row = [b]
        rows.append(sorted(curr_row, key=lambda x: x[0]))

    student_answers = {}
    q_no = 1
    
    # Process each row
    for row in rows:
        if not row: continue
        
        # Segment row into questions based on horizontal gaps
        # Bubbles within a question are close; questions are separated by larger gaps
        questions_in_row = []
        if row:
            curr_q = [row[0]]
            # Calculate average gap between bubbles to define a 'normal' spacing
            all_gaps = [row[i+1][0] - row[i][0] for i in range(len(row)-1)]
            avg_gap = sum(all_gaps)/len(all_gaps) if all_gaps else 100
            
            for i in range(1, len(row)):
                gap = row[i][0] - row[i-1][0]
                # If gap is 1.8x larger than average, it's probably a new question
                if gap > 1.8 * avg_gap:
                    questions_in_row.append(curr_q)
                    curr_q = [row[i]]
                else:
                    curr_row_x_gap = row[i][0] - curr_q[0][0]
                    # Also cap by bubble count (usually 4 or 5)
                    if len(curr_q) < 4:
                        curr_q.append(row[i])
                    else:
                        questions_in_row.append(curr_q)
                        curr_q = [row[i]]
            questions_in_row.append(curr_q)

        for group in questions_in_row:
            # Pad or trim group to exactly 4 options if it's close
            if len(group) < 4: continue # Too few bubbles detected for this question
            if len(group) > 4: group = group[:4]

            pixel_vals = []
            for (x, y, r) in group:
                mask = np.zeros(thresh.shape, dtype="uint8")
                # Expand mask slightly to catch "outside" shading (up to 100% of radius)
                cv2.circle(mask, (x, y), int(r * 0.95), 255, -1)
                pixel_vals.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))

            if not pixel_vals: continue
            
            # Robust selection:
            # 1. Identify the bubble with the most filled pixels
            sorted_vals = sorted(enumerate(pixel_vals), key=lambda x: x[1], reverse=True)
            max_idx, max_val = sorted_vals[0]
            second_max_val = sorted_vals[1][1] if len(sorted_vals) > 1 else 0
            
            q_label = f"Q{q_no}"
            
            # If the most filled bubble is above a minimum threshold AND 
            # significantly more filled than the second best, it's a clear choice.
            if max_val > 50:
                # If the top bubble is at least 1.6x more filled than the next best 
                # (OR the next best is very low), it's a valid single selection.
                if max_val > 1.6 * second_max_val or (max_val > 100 and second_max_val < 60):
                    student_answers[q_label] = chr(65 + max_idx)
                else:
                    # If two bubbles are both highly filled and close in value, it's INVALID
                    if second_max_val > 0.7 * max_val:
                        student_answers[q_label] = "INVALID"
                    else:
                        # Fallback for messy but clear marking
                        student_answers[q_label] = chr(65 + max_idx)
            else:
                student_answers[q_label] = "BLANK"
            
            q_no += 1

    return student_answers


def process_omr(image_path, output_path):
    """
    Interface for app.py to process an OMR sheet.
    Returns: score (always 0 here), total_questions, selected_answers_dict, output_path
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0, 0, {}, ""

    # Capture choices
    student_answers = collect_selected_answers(image)
    
    if "error" in student_answers:
        return 0, 0, {}, ""

    # Save a processed version of the image (the detected answer box)
    answer_box = get_answer_box(image)
    cv2.imwrite(output_path, answer_box)

    total_q = len(student_answers)
    return 0, total_q, student_answers, output_path


def extract_answers(image_path):
    """
    Interface for app.py to extract answers for an answer key from an image.
    Returns: {question_number: option}
    """
    image = cv2.imread(image_path)
    if image is None:
        return {}

    student_answers = collect_selected_answers(image)
    if "error" in student_answers:
        return {}

    # Convert to numeric format for answer keys
    extracted = {}
    for k, v in student_answers.items():
        try:
            if isinstance(k, int):
                extracted[k] = v
            elif isinstance(k, str) and k.startswith("Q"):
                q_num = int(k[1:])
                extracted[q_num] = v
        except Exception:
            continue
    return extracted


# -----------------------------
# APP ROUTE (CAPTURE ONLY)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        answer_key_str = request.form.get("answer_key", "")
        
        if not file or not answer_key_str:
            return "Please provide both an OMR sheet and an answer key."

        # Parse answer key (comma separated)
        answer_key = [a.strip().upper() for a in answer_key_str.split(",") if a.strip()]

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        image = cv2.imread(path)
        
        # Capture choices
        student_answers = collect_selected_answers(image)

        if "error" in student_answers:
             return f"Error: {student_answers['error']}"

        # Evaluation Logic
        score = 0
        total_questions = len(answer_key)
        
        # Build styled results
        results_list_html = ""
        for i, correct_ans in enumerate(answer_key):
            q_label = f"Q{i+1}"
            student_ans = student_answers.get(q_label, "BLANK")
            is_correct = (student_ans == correct_ans)
            
            icon = "✅" if is_correct else "❌"
            color = "#10b981" if is_correct else "#ef4444"
            if student_ans == "BLANK": color = "#94a3b8"; icon = "⚪"
            if student_ans == "INVALID": color = "#f59e0b"; icon = "⚠️"

            if is_correct: score += 1
            
            results_list_html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {color};">
                <div>
                    <span style="color: #94a3b8; font-size: 0.8rem;">{q_label}</span>
                    <div style="font-weight: 500;">Selected: <b>{student_ans}</b></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">Expected: {correct_ans}</div>
                    <div style="color: {color}; font-weight: 600;">{icon}</div>
                </div>
            </div>
            """

        response_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Results - OMR Pro</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
            <style>
                body {{ font-family: 'Outfit', sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; padding: 40px 20px; margin: 0; }}
                .container {{ background: rgba(255,255,255,0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); padding: 30px; border-radius: 24px; width: 100%; max-width: 500px; }}
                h2 {{ text-align: center; background: linear-gradient(to right, #818cf8, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .score-card {{ text-align: center; padding: 20px; background: rgba(79, 70, 229, 0.1); border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(79, 70, 229, 0.2); }}
                .score-val {{ font-size: 2.5rem; font-weight: 600; color: #818cf8; }}
                .btn {{ display: block; text-align: center; padding: 14px; background: #4f46e5; color: white; text-decoration: none; border-radius: 12px; font-weight: 600; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Evaluation Results</h2>
                <div class="score-card">
                    <div style="font-size: 0.9rem; color: #94a3b8;">Final Score</div>
                    <div class="score-val">{score} / {total_questions}</div>
                </div>
                <div>{results_list_html}</div>
                <a href="/" class="btn">Scan Another Sheet</a>
            </div>
        </body>
        </html>
        """
        return response_html

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)