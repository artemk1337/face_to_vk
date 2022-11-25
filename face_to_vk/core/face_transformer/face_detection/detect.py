from typing import List
from mtcnn.mtcnn import MTCNN
# from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2


class FaceDetector:
    FACE_DETECTOR = MTCNN()
    TARGET_SIZE = (160, 160)

    def __init__(self, confidence: float = 0.98):
        self.confidence = confidence

    # MTCNN tensorflow
    def _box_to_array(self, img: np.array, x: int, y: int, width: int, height: int) -> np.array:
        """
        Create face array from box and src img

        :param img: src image with faces
        :param x: x
        :param y: y
        :param width: width shift
        :param height: heigh shift
        :return: face array
        """
        face = img[y:y+height, x:x+width]
        if self.TARGET_SIZE:
            face = cv2.resize(face, self.TARGET_SIZE)
        return face

    def detect(self, img: np.array) -> list:
        """
        Detect faces on img

        :param img: src image
        :return: list with RGB face arrays
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


if __name__ == "__main__":
    from core.utils.download_img import download_img
    import tensorflow as tf
    import time

    print(tf.test.is_gpu_available())

    tm_start = time.time()

    detector = FaceDetector()
    img = download_img("https://sun1-55.userapi.com/impf/c638631/v638631152/2999e/w2LNzjmVlf4.jpg?size=150x150&quality=96&sign=016171fd8dac67900de27138a230aeb3&type=album")
    faces = detector.detect(img)
    Image.fromarray(faces[0]).save("test1.png")
    print(len(faces), faces[0].shape)

    print(f"Seconds: {(time.time() - tm_start):.2f}")
