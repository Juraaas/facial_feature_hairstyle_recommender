import numpy as np

class FaceGeometry:
    def __init__(self, landmarks, hairline_y=None):
        self.lm = landmarks
        self.hairline_y = hairline_y

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
        #return np.array([np.mean(self.lm[:, 0]), np.min(self.lm[:, 1])])
        if self.hairline_y is not None:
            mid_x = (self._p(234)[0] + self._p(454)[0]) / 2
            return np.array([mid_x, self.hairline_y])
        return self._p(10)
    
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
        mid = (self._p(33) + self._p(263)) / 2
        left_dist = np.linalg.norm(self._p(234) - mid)
        right_dist = np.linalg.norm(self._p(454) - mid)
        return abs(left_dist - right_dist) / self.face_width()
    
    def jaw_to_height(self):
        return self.jaw_width() / self.face_height()
    
    def eye_ratio(self):
        return self.eye_dist() / self.face_width()
    
    def eye_height(self):
        eye_h = self.dist(self._p(159), self._p(145))
        eye_w = self.dist(self._p(33), self._p(133))
        return eye_h / eye_w

    def lip_ratio(self):
        return self.dist(self._p(61), self._p(291)) / self.face_width()

    def chin_ratio(self):
        return self.dist(self._p(152), self._p(175)) / self.face_height()

    def forehead_ratio(self):
        forehead_h = self.dist(self.forehead_top(), self._p(10))
        return forehead_h / self.face_height()
    
    def lower_face_ratio(self):
        mouth = self._p(0)
        return self.dist(mouth, self.chin()) / self.face_height()
    
    def chin_prominence(self):
        jaw_mid = (self._p(172) + self._p(397)) / 2
        return self.dist(jaw_mid, self.chin()) / self.face_height()
    
    def brow_line_y(self):
        left_brow  = self._p(55)
        right_brow = self._p(285)
        return (left_brow[1] + right_brow[1]) / 2
    
    def nose_base_y(self):
        return self._p(2)[1]
    
    def upper_third(self):
        top = self.forehead_top()[1]
        return abs(self.brow_line_y() - top)
    
    def middle_third(self):
        return abs(self.nose_base_y() - self.brow_line_y())
    
    def lower_third(self):
        return abs(self.chin()[1] - self.nose_base_y())
    
    def facial_thirds_ratio(self):
        u = self.upper_third()
        m = self.middle_third()
        l = self.lower_third()
        total = u + m + l
        if total == 0:
            return None
        return {
            "upper": round(u / total, 3),
            "middle": round(m / total, 3),
            "lower": round(l / total, 3),
        }
    
    def middle_lower_ratio(self):
        m = self.middle_third()
        l = self.lower_third()
        return m / l if l > 0 else 1.0