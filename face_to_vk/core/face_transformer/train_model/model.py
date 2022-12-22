from torchvision.models.efficientnet import (
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
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
    def create(batch_size: int = 32, output_size: int = 256, add_ctivation=False):
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

        return model


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)  # 5

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TrainModel:
    # tensorboard --logdir=/home/artem/projects/face_to_vk/face_to_vk/core/face_transformer/face_to_vec/train_model/runs
    TENSORBOARD_LOG_DIR = None
    TENSORBOARD_FLUSH_SECS = 15

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
            async_mode: bool = True,
            loss_fn_name: str = 'triplet',
    ):
        self.model = model
        self.train_data_loader, self.test_data_loader = self._create_data_loaders(train_dir, test_dir)
        self.async_mode = async_mode

        self.writer: Optional[SummaryWriter] = None

        self.loss_fn_name = loss_fn_name
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1)
        self.contrastive_loss = ContrastiveLoss(margin=1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.scheduler_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.75, patience=25)

    def _calc_triplet_loss(self, anchor, positive, negative, backward: bool = True):
        loss = self.triplet_loss(anchor, positive, negative)
        if backward:
            loss.backward()
        return loss

    def _calc_contrastive_loss(self, anchor, positive, negative, backward: bool = True):
        loss1 = self.contrastive_loss(anchor, positive, 0)
        if backward:
            loss1.backward()
        loss2 = self.contrastive_loss(anchor, negative, 1)
        if backward:
            loss2.backward()
        return loss1 + loss2

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
        if self.loss_fn_name == 'triplet':
            loss = self._calc_triplet_loss(anchor, positive, negative, backward=True)
        elif self.loss_fn_name == 'contrastive':
            loss = self._calc_contrastive_loss(anchor, positive, negative, backward=True)
        else:
            raise
        return loss

    def train(self, batch_size: int = 16, epochs: int = 1000, mini_batches: Optional[int] = 1):
        start_train_time = time.time()

        if self.async_mode:
            self.train_data_loader.start_async_reader(queue_size=5, batch_size=batch_size)
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

            # step optimizer
            self.optimizer.step()

            # calc loss
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            total_losses += [mean_loss]
            if self.writer:
                self.writer.add_scalar('Loss/train', mean_loss, epoch)
            print(f"Epoch: {epoch}; "
                  f"Mean loss: {mean_loss}; "
                  f"Time epoch: {time.time() - time_epoch:.2f}; "
                  f"Optimizer step: {self.optimizer.param_groups[0]['lr']}")

            # update step optimizer
            # self.scheduler_optimizer.step(mean_loss)

            # save model1.weights
            if mean_loss == min(total_losses):
                torch.save(self.model.state_dict(), "../model1.weights")

        # finish
        print(f"Total time: {time.time() - start_train_time:.2f} seconds.")

        self.train_data_loader.stop_async_reader()


if __name__ == "__main__":
    output_size = 256
    epochs = 3000
    batch_size = 16
    mini_batches = 100
    loss_fn_name = 'triplet'

    cnn_model = Model.create(batch_size=batch_size, output_size=output_size)
    cnn_model.load_state_dict(torch.load("../model1.weights"))

    train_model = TrainModel(model=cnn_model, train_dir=TRAIN_DIR, test_dir=TEST_DIR, async_mode=True)
    train_model.create_tensorboard(
        comment=f" efficientnet_v2_m, pretrained false, 1 linear layers, no activation, loss name {loss_fn_name},"
                f" output_size {output_size}, epochs {epochs}, batch_size {batch_size}, mini_batches {mini_batches}")
    train_model.train(epochs=epochs, mini_batches=mini_batches, batch_size=batch_size)

    # torch.save(cnn_model.state_dict(), "model1.weights")
