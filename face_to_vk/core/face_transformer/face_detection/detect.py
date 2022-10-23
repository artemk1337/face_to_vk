from typing import Optional, Union, List
from mtcnn import MTCNN
import numpy as np


class FaceDetector:
    FACE_DETECTOR = MTCNN()

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    @staticmethod
    def _box_to_array(img: np.array, x: int, y: int, width: int, height: int) -> np.array:
        """
        Create face array from box and src img

        :param img: src image with faces
        :param x: x
        :param y: y
        :param width: width shift
        :param height: heigh shift
        :return: face array
        """
        return img[y:y+height][x:x+width]

    def detect(self, img: np.array) -> list:
        """
        Detect faces on img

        :param img: src image
        :return: list with face arrays
        """
        faces: List[dict] = self.FACE_DETECTOR.detect_faces(img)
        """
        Example faces:
        [
            {
                'box': [277, 90, 48, 63],
                'keypoints':
                {
                    'nose': (303, 131),
                    'mouth_right': (313, 141),
                    'right_eye': (314, 114),
                    'left_eye': (291, 117),
                    'mouth_left': (296, 143)
                },
                'confidence': 0.99851983785629272
            }
        ]
        """
        return [self._box_to_array(img, *face['box']) for face in faces if face['confidence'] >= self.confidence]
