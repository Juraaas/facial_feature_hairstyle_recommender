import numpy as np

class FaceGeometry:
    def __init__(self, landmarks):
        self.lm = landmarks

    def _p(self, idx):
        return self.lm[idx][:2]
    
    def chin(self):
        return self._p(152)
    
    def nose(self):
        return self._p(1)
    
    def left_cheek(self):
        return self._p(234)

    def right_cheek(self):
        return self._p(454)

    def left_eye(self):
        return self._p(33)

    def right_eye(self):
        return self._p(263)

    def forehead_top(self):
        return np.array([np.mean(self.lm[:, 0]), np.min(self.lm[:, 1])])
    
    def dist(self, a, b):
        return np.linalg.norm(a - b)
    
    def face_height(self):
        return self.dist(self.chin(), self.forehead_top())
    
    def face_width(self):
        return self.dist(self.left_cheek(), self.right_cheek())
    
    def jaw_width(self):
        return self.dist(self._p(172), self._p(397))
    
    def eye_dist(self):
        return self.dist(self.left_eye(), self.right_eye())
    
    def nose_position_ratio(self):
        return (self.nose()[1] - self.forehead_top()[1]) / self.face_height()
    
    def face_ratio(self):
        return self.face_height() / self.face_width()
    
    def jaw_ratio(self):
        return self.jaw_width() / self.face_width()
    
    def symmetry_score(self):
        left = self._p(234)
        right = self._p(454)
        mid = self.nose()

        left_dist = np.linalg.norm(left - mid)
        right_dist = np.linalg.norm(right - mid)

        return abs(left_dist - right_dist) / self.face_width()
    
    def jaw_to_height(self):
        return self.jaw_width() / self.face_height()
    
    def eye_ratio(self):
        return self.eye_dist() / self.face_width()
    
    def jaw_projection(self):
        chin = self.chin()
        nose = self.nose()
        proj = self.dist(chin, nose)

        return proj / self.face_height()