from torchvision.models.efficientnet import (
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
import timm

from torchinfo import summary
from torchvision.models.resnet import resnet50
import torch.nn.functional as F
import torch

TRAIN_DIR, TEST_DIR = '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_train/train_processed_160', \
    '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test_processed_160'

device = "cuda" if torch.cuda.is_available() else "cpu"


class ResNet50:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = timm.create_model('resnet50', pretrained=pretrained)
        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=2048,
                            out_features=output_size,
                            bias=True),
        ).to(device)
        model.eval()
        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model


class InceptionV3:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = timm.create_model('inception_v3', pretrained=pretrained)
        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=2048,
                            out_features=output_size,
                            bias=True),
        ).to(device)
        model.eval()
        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model


class InceptionV4:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = timm.create_model('inception_v4', pretrained=pretrained)
        model.fc = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=2048,
                            out_features=output_size,
                            bias=True),
        ).to(device)
        model.eval()
        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model


class InceptionResnetV2:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = timm.create_model('inception_resnet_v2', pretrained=pretrained)
        model.classif = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=1536,
                            out_features=output_size,
                            bias=True),
        ).to(device)
        model.eval()
        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model


class EfficientNetV2S:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights if pretrained else None).to(device)
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

        return model


class EfficientNetV2M:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights if pretrained else None).to(device)
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

        return model


class EfficientNetB7:
    @classmethod
    def create(cls, batch_size: int = 32, output_size: int = 256, pretrained: bool = True):
        model = timm.create_model('efficientnet_b7', pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(0.2),
            # torch.nn.Linear(in_features=1280,
            #                 out_features=512,
            #                 bias=True),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=2560,
                            out_features=output_size,
                            bias=True),
            # torch.nn.Tanh(),
        ).to(device)

        model.eval()
        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return model
