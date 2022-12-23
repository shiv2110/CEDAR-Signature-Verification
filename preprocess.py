import cv2

def preprocess (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(gray, (155, 220), interpolation = cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(img1, 220, 255, 0)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return thresh

# preprocess()