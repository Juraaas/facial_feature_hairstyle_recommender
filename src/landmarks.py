import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceLandmarkDetector:
    def __init__(self, model_path):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.landmarker =  FaceLandmarker.create_from_options(options)
    
    def detect(self, image):
        h, w, _ = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None
        
        landmarks = result.face_landmarks[0]

        coords = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            z = lm.z
            coords.append([x, y, z])

        return np.array(coords)