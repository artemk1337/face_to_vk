from torchvision.models.efficientnet import (
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
from torchinfo import summary
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


class Face2VecModel:
    def __init__(self):
        self.model = None

    def create(self, batch_size: int = 32, output_size: int = 256):
        model = efficientnet_v2_m().to(device)
        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(in_features=1280,
            #                 out_features=512,
            #                 bias=True),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1280,
                            out_features=output_size,
                            bias=True),
            # torch.nn.Tanh(),
        ).to(device)

        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )

        self.model = model
        return self

    def load_weights(self, path_to_weights: str = 'model1.weights'):
        self.model.load_state_dict(torch.load(path_to_weights))
        return self

    def transform(self, batch_imgs):
        assert batch_imgs.shape[0] > 0 and batch_imgs.shape[1] == 3 and batch_imgs.shape[2] == 160 and batch_imgs.shape[3] == 160
        return self.model(batch_imgs)


# FACE2VEC_MODEL = Model().create()
# FACE2VEC_MODEL.load_weights('model1.weights')
