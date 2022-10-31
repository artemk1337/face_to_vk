from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch


class FaceTransformer:
    FACE_TRANSFORMER = InceptionResnetV1(pretrained='vggface2').eval().\
        to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    @classmethod
    def transform(cls, img: np.array):
        res = cls.FACE_TRANSFORMER.forward(img)
        print(res)
        return res


if __name__ == "__main__":
    from core.face_transformer.face_detection.detect import FaceDetector
    from core.utils.download_img import download_img
    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)
    detector = FaceDetector()
    img = download_img(
        "https://sun9-81.userapi.com/impg/"
        "QEXlCUZnqfYl6bVrT-B819Qk1vtimz29Lagnww/"
        "JuWBdTp9Jx4.jpg?size=2560x1920&quality=95&sign=531d83edab642e441ec55325d991ad29&type=album")
    faces = detector.detect(img)
    print(len(faces), faces[0].shape)
    res = FaceTransformer.transform(faces[0])
    print(res)
