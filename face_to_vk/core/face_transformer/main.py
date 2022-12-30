from typing import Union
from core.face_transformer.mtcnn import FaceDetector
from core.face_transformer.face_to_vec import Face2VecModel
from core.face_transformer.models import InceptionResnetV2


class FaceTransformer:

    def __init__(self):
        self.mtcnn = FaceDetector()
        self.model_face2vec = Face2VecModel().\
            create(batch_size=32, output_size=512, class_model=InceptionResnetV2).\
            load_weights('inception_resnet_v2_512_0.867.weights')

    def transform(self, imgs):
        faces = self.mtcnn.detect(imgs)  # shape [batch_size, 3, 160, 160]
        vecs = self.model_face2vec.transform(faces)
        print(vecs)
        return vecs


if __name__ == "__main__":
    from core.utils.download_img import download_img

    face_transformer = FaceTransformer()
    img = download_img("https://sun9-52.userapi.com/impf/c858420/v858420603/77d2a/1ZZNJXmifbw.jpg?size=2560x1920&quality=96&sign=2f5e864d9e32c8bacb5da5225ffe55e4&type=album")
    faces = face_transformer.transform([img, img])
