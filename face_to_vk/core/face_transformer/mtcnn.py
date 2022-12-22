from typing import Optional

from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
import torch

import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FaceDetector:
    TARGET_SIZE = (160, 160)
    FACE_DETECTOR = MTCNN(image_size=TARGET_SIZE[0], keep_all=True, device=device, post_process=False)
    TORCH_TRANSFORM = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        # transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    def __init__(self, confidence: float = 0.99):
        self.confidence = confidence

    def detect(self, imgs: np.array, min_prob: float = 0.9) -> Optional[torch.Tensor]:
        """
        Detect faces on img

        :param imgs: src images
        :param min_prob:
        :return: list with RGB face arrays
        """
        # faces: List[dict] = self.FACE_DETECTOR.detect(img)
        faces, probs = self.FACE_DETECTOR(imgs, return_prob=True)
        # faces shape [images, faces, 3, 160, 160]
        # probs shape [images, faces]
        batch = []
        for imgs, img_probs in zip(faces, probs):
            # imgs shape [n, 3, 160, 160], n - faces
            # img_probs [n]
            for face, img_prob in zip(imgs, img_probs):
                if img_prob < min_prob:
                    continue
                face = self.TORCH_TRANSFORM(face)
                face = face[None, :]
                batch += [face]
        return torch.cat(batch).to(device) if batch else None


if __name__ == "__main__":
    from core.utils.download_img import download_img
    import time

    tm_start = time.time()

    detector = FaceDetector()
    img = download_img("https://sun9-52.userapi.com/impf/c858420/v858420603/77d2a/1ZZNJXmifbw.jpg?size=2560x1920&quality=96&sign=2f5e864d9e32c8bacb5da5225ffe55e4&type=album")
    faces = detector.detect([img, img, img])
    print(len(faces), faces.shape)

    print(f"Seconds: {(time.time() - tm_start):.2f}")
