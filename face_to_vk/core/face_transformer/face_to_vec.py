import torch

from core.face_transformer.models import InceptionResnetV2


device = "cuda" if torch.cuda.is_available() else "cpu"


class Face2VecModel:
    def __init__(self):
        self.model = None

    def create(self, batch_size: int = 32, output_size: int = 512, class_model=InceptionResnetV2):
        self.model = class_model.create(batch_size=batch_size, output_size=output_size)
        return self

    def load_weights(self, path_to_weights: str):
        self.model.load_state_dict(torch.load(path_to_weights))
        return self

    def transform(self, batch_imgs):
        assert batch_imgs.shape[0] > 0 and \
               batch_imgs.shape[1] == 3 and \
               batch_imgs.shape[2] == 160 and \
               batch_imgs.shape[3] == 160
        return self.model(batch_imgs)


# FACE2VEC_MODEL = Model().create()
# FACE2VEC_MODEL.load_weights('best_model_eff_m.weights')
