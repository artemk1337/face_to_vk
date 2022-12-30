import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import timm

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch

from tqdm import tqdm
import time
import os

from typing import Optional

from core.face_transformer.train_model.prepare_data import CustomDataLoader
from core.face_transformer.models import *


TRAIN_DIR, TEST_DIR = '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_train/train_processed_160', \
    '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test_processed_160'

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    MODEL_NAME = None
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
            lr: float = 0.001,
            lr_scheduler: bool = False
    ):
        self.model = model
        self.train_data_loader, self.test_data_loader = self._create_data_loaders(train_dir, test_dir)
        self.async_mode = async_mode

        self.writer: Optional[SummaryWriter] = None

        self.loss_fn_name = loss_fn_name
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1)
        self.contrastive_loss = ContrastiveLoss(margin=1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = lr_scheduler
        if lr_scheduler:
            self.scheduler_optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.75, patience=50)

    def _calc_triplet_loss(self, anchor, positive, negative, backward: bool = True, norm_k: Optional[int] = None):
        loss = self.triplet_loss(anchor, positive, negative)
        if backward:
            loss.backward()
        if norm_k:
            loss /= norm_k
        return loss

    def _calc_contrastive_loss(self, anchor, positive, negative, backward: bool = True, norm_k: Optional[int] = None):
        loss1 = self.contrastive_loss(anchor, positive, 0)
        loss2 = self.contrastive_loss(anchor, negative, 1)
        total_loss = loss1 + loss2
        if norm_k:
            total_loss /= norm_k
        if backward:
            total_loss.backward()

        return total_loss

    def get_batch(self, data_loader: CustomDataLoader, batch_size: int):
        if self.async_mode and data_loader.queue:
            return data_loader.queue.get()
        return data_loader.create_triplet_batch(batch_size)

    def create_tensorboard(self, comment: str = "", purge_step: Optional[int] = None):
        self.writer = SummaryWriter(log_dir=self.TENSORBOARD_LOG_DIR, comment=comment,
                                    flush_secs=self.TENSORBOARD_FLUSH_SECS, purge_step=purge_step)

    def predict(self, data):
        return self.model(data)

    def _predict_and_calc_loss_on_train(self, batch_size: int, norm_k: Optional[int] = None):
        anchor, positive, negative = self.get_batch(self.train_data_loader, batch_size)
        anchor, positive, negative = self.predict(anchor), self.predict(positive), self.predict(negative)
        if self.loss_fn_name == 'triplet':
            loss = self._calc_triplet_loss(anchor, positive, negative, backward=True, norm_k=norm_k)
        elif self.loss_fn_name == 'contrastive':
            loss = self._calc_contrastive_loss(anchor, positive, negative, backward=True, norm_k=norm_k)
        else:
            raise ValueError("Not correct loss function name")
        return loss

    def _predict_and_calc_score_on_test(self, batch_size: int, mini_batches: int = 1):
        total_losses_pos = []
        total_losses_neg = []

        for mini_batch in range(mini_batches):
            anchor, positive, negative = self.get_batch(self.test_data_loader, batch_size)
            anchor, positive, negative = self.predict(anchor), self.predict(positive), self.predict(negative)
            euclidean_distance_pos = F.pairwise_distance(anchor, positive, keepdim=True).cpu().detach().numpy().squeeze()
            euclidean_distance_neg = F.pairwise_distance(anchor, negative, keepdim=True).cpu().detach().numpy().squeeze()
            euclidean_distance_pos: list = np.clip(euclidean_distance_pos, 0, 1).tolist()
            euclidean_distance_neg: list = np.clip(euclidean_distance_neg, 0, 1).tolist()
            total_losses_pos += euclidean_distance_pos
            total_losses_neg += euclidean_distance_neg

        total_targets = np.zeros(len(total_losses_pos)).tolist() + np.ones(len(total_losses_neg)).tolist()
        total_losses = total_losses_pos + total_losses_neg

        roc_auc = roc_auc_score(total_targets, total_losses, average='weighted')
        acc = accuracy_score(total_targets, np.round(total_losses))
        return acc, roc_auc

    def train(self, batch_size: int = 16, epochs: int = 1000, mini_batches: Optional[int] = 1):
        start_train_time = time.time()

        if self.async_mode:
            self.train_data_loader.start_async_reader(queue_size=5, batch_size=batch_size)
            self.test_data_loader.start_async_reader(queue_size=2, batch_size=batch_size)
        if self.writer:
            self.writer.add_graph(self.model, self.get_batch(self.train_data_loader, batch_size=1)[0])

        total_losses = []

        for epoch in range(epochs):

            epoch_losses = []
            time_epoch = time.time()
            self.optimizer.zero_grad()

            for mini_batch in tqdm(range(mini_batches)):
                loss = self._predict_and_calc_loss_on_train(batch_size=batch_size, norm_k=mini_batches)
                epoch_losses += [loss.item()]  # normilized 1 / mini_batches

            # step optimizer
            self.optimizer.step()

            # calc loss
            mean_loss = sum(epoch_losses)
            total_losses += [mean_loss]

            test_acc, test_auc_roc = self._predict_and_calc_score_on_test(
                batch_size=batch_size,
                mini_batches=max(mini_batches // 10, 1))
            if self.writer:
                self.writer.add_scalar('Loss/train', mean_loss, epoch)
                self.writer.add_scalar('test/accuracy', test_acc, epoch)
                self.writer.add_scalar('test/auc-roc', test_auc_roc, epoch)
            print(f"Epoch: {epoch}; "
                  f"Mean loss: {mean_loss}; "
                  f"Time epoch: {time.time() - time_epoch:.2f}; "
                  f"Optimizer step: {self.optimizer.param_groups[0]['lr']}; "
                  f"Validation metrics: acc - {test_acc:.2f}, auc-roc - {test_auc_roc:.2f}")

            # update step optimizer
            if self.lr_scheduler:
                self.scheduler_optimizer.step(mean_loss)

            # save best_model_eff_m.weights
            if mean_loss == min(total_losses):
                torch.save(self.model.state_dict(), f"../{self.MODEL_NAME}.weights")

        # finish
        print(f"Total time: {time.time() - start_train_time:.2f} seconds.")

        self.train_data_loader.stop_async_reader()


def train(model_name=None, model_class=None, loss_fn=None, lr=None, purge_step=None):
    output_size = 512
    epochs = 5000
    batch_size = 32
    mini_batches = 100
    loss_fn_name = loss_fn or 'contrastive'
    preload_local_weights = False
    pretrained = True
    lr = lr or 0.0001
    purge_step = purge_step or None

    name = model_name or 'InceptionV3'
    cnn_model = (model_class or InceptionV3).create(batch_size=batch_size, output_size=output_size, pretrained=pretrained)
    weights_path = f"../{name}.weights"
    if os.path.exists(weights_path):
        preload_local_weights = True
        cnn_model.load_state_dict(torch.load(weights_path))

    TrainModel.MODEL_NAME = name
    train_model = TrainModel(model=cnn_model, train_dir=TRAIN_DIR, test_dir=TEST_DIR, async_mode=True,
                             loss_fn_name=loss_fn_name, lr=lr)
    train_model.create_tensorboard(
        comment=f" {name}, pretrained {pretrained}, 1 linear layers, no activation,"
                f" loss name {loss_fn_name},"
                f" output_size {output_size}, epochs {epochs}, batch_size {batch_size}, mini_batches {mini_batches},"
                f" preload local weights {preload_local_weights}, lr {lr}", purge_step=purge_step)
    train_model.train(epochs=epochs, mini_batches=mini_batches, batch_size=batch_size)

    # torch.save(cnn_model.state_dict(), "best_model_eff_m.weights")


if __name__ == "__main__":
    # for model_name, model_class in zip(
    #         ('InceptionV3', 'InceptionV4', 'InceptionResnetV2'),
    #         (InceptionV3, InceptionV4, InceptionResnetV2)
    # ):
    #     # for loss_fn in ('contrastive', 'triplet'):
    #     train(model_name=model_name, model_class=model_class)
    train(model_name='InceptionResnetV2', model_class=InceptionResnetV2, lr=0.001, loss_fn='triplet')
