from typing import Optional
from mtcnn import MTCNN


class FaceDetector:
    detector = MTCNN()

    def __init__(self, imgs: list):
        pass

    def detect(self, imgs: Optional[list] = None):
        for img in imgs:
            self.detector.detect_faces(img)
