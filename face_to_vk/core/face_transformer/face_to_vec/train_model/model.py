from torchvision.models.efficientnet import (
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch

from tqdm import tqdm
import time

from typing import Optional

from prepare_data import CustomDataLoader


TRAIN_DIR, TEST_DIR = '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_train/train_processed_160', \
                      '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test_processed_160'


device = "cuda" if torch.cuda.is_available() else "cpu"


class Model:
    def __init__(self):
        pass

    @staticmethod
    def create(batch_size: int = 32):
        model = efficientnet_v2_s(pretrained=True, weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=1280,
                            out_features=128,
                            bias=True),
            torch.nn.Tanh(),
        ).to(device)

        summary(model=model,
                input_size=(batch_size, 3, 160, 160),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )

        return model


class TrainModel:

    # tensorboard --logdir=/home/artem/projects/face_to_vk/face_to_vk/core/face_transformer/face_to_vec/train_model/runs
    TENSORBOARD_LOG_DIR = None
    TENSORBOARD_FLUSH_SECS = 10

    @classmethod
    def _create_data_loaders(cls, train_dir: str, test_dir: str = None):
        train_data_loader = CustomDataLoader(train_dir)
        test_data_loader = CustomDataLoader(test_dir) if test_dir else None
        return train_data_loader, test_data_loader

    def __init__(
            self,
            model: torch.nn.Module,
            train_dir: str,
            test_dir: str,
            async_mode: bool = True
    ):
        self.model = model
        self.train_data_loader, self.test_data_loader = self._create_data_loaders(train_dir, test_dir)
        self.async_mode = async_mode

        self.writer: Optional[SummaryWriter] = None

        self.loss_fn = torch.nn.TripletMarginLoss(margin=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _calc_triplet_loss(self, anchor, positive, negative, bakward: bool = True):
        loss = self.loss_fn(anchor, positive, negative)
        if bakward:
            loss.backward()
        return loss

    def get_batch(self, data_loader: CustomDataLoader, batch_size: int):
        if self.async_mode and data_loader.queue:
            return data_loader.queue.get()
        return data_loader.create_triplet_batch(batch_size)

    def create_tensorboard(self, comment: str = ""):
        self.writer = SummaryWriter(log_dir=self.TENSORBOARD_LOG_DIR, comment=comment,
                                    flush_secs=self.TENSORBOARD_FLUSH_SECS)

    def predict(self, data):
        return self.model(data)

    def _predict_and_calc_loss(self, batch_size: int):
        anchor, positive, negative = self.get_batch(self.train_data_loader, batch_size)
        anchor, positive, negative = self.predict(anchor), self.predict(positive), self.predict(negative)
        loss = self._calc_triplet_loss(anchor, positive, negative, bakward=True)
        return loss

    def train(self, batch_size: int = 16, epochs: int = 1000, mini_batches: Optional[int] = 1):
        start_train_time = time.time()

        if self.async_mode:
            self.train_data_loader.start_async_reader(batch_size=batch_size)
        if self.writer:
            self.writer.add_graph(self.model, self.get_batch(self.train_data_loader, batch_size=1)[0])

        total_losses = []

        for epoch in range(epochs):

            epoch_losses = []
            time_epoch = time.time()
            self.optimizer.zero_grad()

            for mini_batch in tqdm(range(mini_batches)):
                loss = self._predict_and_calc_loss(batch_size=batch_size)
                epoch_losses += [loss.item()]

            self.optimizer.step()

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            total_losses += [mean_loss]
            if self.writer:
                self.writer.add_scalar('Loss/train', mean_loss, epoch)
            print(f"Epoch: {epoch}; Mean loss: {mean_loss}; Time epoch: {time.time() - time_epoch:.2f}")

        print(f"Total time: {time.time() - start_train_time:.2f} seconds.")

        self.train_data_loader.stop_async_reader()


if __name__ == "__main__":

    batch_size = 32

    cnn_model = Model.create(batch_size=batch_size)

    train_model = TrainModel(model=cnn_model, train_dir=TRAIN_DIR, test_dir=TEST_DIR, async_mode=True)
    train_model.create_tensorboard(comment="efficientnet_v2_s, output 128 units")
    train_model.train(epochs=1000, mini_batches=50, batch_size=batch_size)
