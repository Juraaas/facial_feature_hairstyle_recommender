import cv2
from src.landmarks import FaceLandmarkDetector

img = cv2.imread("dataset/test_images/004455.jpg")

detector = FaceLandmarkDetector(
    model_path="models/face_landmarker.task"
)
landmarks = detector.detect(img)

if landmarks is None:
    print("No face detected")
    exit()

img = detector.draw_landmarks(img, landmarks, draw_indices=False)
img = cv2.resize(img, None, fx=2, fy=2)

cv2.imshow("Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()