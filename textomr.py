import cv2
import numpy as np
import imutils

# -----------------------------
# STEP 1: LOAD IMAGE
# -----------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Error: Unable to load image.")
    return img


# -----------------------------
# STEP 2: CHECK IF IMAGE IS OMR
# -----------------------------
def is_omr_sheet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    # Find largest contour (should be paper)
    largest = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    # OMR sheet should have 4 corners (rectangle)
    if len(approx) == 4:
        return True

    return False


# -----------------------------
# STEP 3: PERSPECTIVE TRANSFORM
# -----------------------------
def four_point_transform(image, pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# -----------------------------
# STEP 4: DETECT AND ALIGN SHEET
# -----------------------------
def get_warped_sheet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return four_point_transform(image, approx)

    raise Exception("Error: Could not detect OMR sheet.")


# -----------------------------
# STEP 5: PROCESS OMR
# -----------------------------
def evaluate_omr(image, total_questions=20, options=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_contours = []

    # Filter circular bubbles
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)

        if 15 <= w <= 50 and 15 <= h <= 50 and 0.8 <= aspect_ratio <= 1.2:
            bubble_contours.append(c)

    # Sort top to bottom
    bubble_contours = imutils.contours.sort_contours(bubble_contours, method="top-to-bottom")[0]

    answers = {}
    question_index = 0

    # Process each question
    for i in range(0, len(bubble_contours), options):
        cnts = bubble_contours[i:i + options]

        cnts = imutils.contours.sort_contours(cnts, method="left-to-right")[0]

        filled = []
        pixel_counts = []

        for j, c in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            pixel_counts.append(total)

        max_val = max(pixel_counts)

        # Detect selected option
        selected = [idx for idx, val in enumerate(pixel_counts) if val > 0.8 * max_val]

        question_index += 1

        if len(selected) == 1:
            answers[f"Q{question_index}"] = chr(65 + selected[0])
        elif len(selected) > 1:
            answers[f"Q{question_index}"] = "INVALID"
        else:
            answers[f"Q{question_index}"] = "BLANK"

    return answers


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def process_omr(path):
    try:
        image = load_image(path)

        # Step 2: Check OMR
        if not is_omr_sheet(image):
            return "Error: The given image is NOT a valid OMR sheet."

        # Step 3: Align
        warped = get_warped_sheet(image)

        # Step 4: Evaluate
        result = evaluate_omr(warped)

        return result

    except Exception as e:
        return str(e)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    path = input("Enter OMR image path: ")
    output = process_omr(path)

    print("\nResult:")
    print(output)