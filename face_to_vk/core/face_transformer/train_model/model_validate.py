import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score

import time
import os
from tqdm import tqdm

from typing import Optional

from core.face_transformer.models import *
from prepare_data import CustomDataLoader

TRAIN_DIR, TEST_DIR = '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_train/train_processed_160', \
    '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test_processed_160'

device = "cuda" if torch.cuda.is_available() else "cpu"


class ValidateModel:

    @classmethod
    def _create_data_loader(cls, test_dir: str = None):
        test_data_loader = CustomDataLoader(test_dir) if test_dir else None
        return test_data_loader

    def __init__(
            self,
            model: torch.nn.Module,
            test_dir: str,
            async_mode: bool = True,
    ):
        self.model = model
        self.test_data_loader = self._create_data_loader(test_dir)
        self.async_mode = async_mode

    def get_batch(self, data_loader: CustomDataLoader, batch_size: int):
        if self.async_mode and data_loader.queue:
            return data_loader.queue.get()
        return data_loader.create_triplet_batch(batch_size)

    def predict(self, data):
        return self.model(data)

    def _predict_and_calc_loss(self, batch_size: int) -> (list, list):
        anchor, positive, negative = self.get_batch(self.test_data_loader, batch_size)
        anchor, positive, negative = self.predict(anchor), self.predict(positive), self.predict(negative)
        euclidean_distance_pos = F.pairwise_distance(anchor, positive, keepdim=True).cpu().detach().numpy().squeeze()
        euclidean_distance_neg = F.pairwise_distance(anchor, negative, keepdim=True).cpu().detach().numpy().squeeze()
        euclidean_distance_pos: list = np.clip(euclidean_distance_pos, 0, 1).tolist()
        euclidean_distance_neg: list = np.clip(euclidean_distance_neg, 0, 1).tolist()
        return euclidean_distance_pos, euclidean_distance_neg

    def validate(self, batch_size: int = 16, epochs: int = 1000, mini_batches: Optional[int] = 1):
        start_test_time = time.time()

        if self.async_mode:
            self.test_data_loader.start_async_reader(queue_size=5, batch_size=batch_size)

        total_losses = []
        total_targets = []

        for epoch in range(epochs):
            time_epoch = time.time()

            for mini_batch in tqdm(range(mini_batches)):
                pos_prob, neg_prob = self._predict_and_calc_loss(batch_size=batch_size)
                total_losses += pos_prob
                total_targets += np.zeros(len(pos_prob)).tolist()
                total_losses += neg_prob
                total_targets += np.ones(len(neg_prob)).tolist()

        print(f'\nResults on {int(epochs*mini_batches*batch_size)} samples')
        print(roc_auc_score(total_targets, total_losses, average='weighted'))
        print(accuracy_score(total_targets, np.round(total_losses)))
        self.test_data_loader.stop_async_reader()


if __name__ == "__main__":
    output_size = 512
    epochs = 10
    batch_size = 8
    mini_batches = 20

    cnn_model = InceptionResnetV2.create(batch_size=batch_size, output_size=output_size)
    weights_path = f"../inception_resnet_v2_512_0.867.weights"
    if os.path.exists(weights_path):
        cnn_model.load_state_dict(torch.load(weights_path))

    train_model = ValidateModel(model=cnn_model, test_dir=TEST_DIR, async_mode=True)
    train_model.validate(epochs=epochs, mini_batches=mini_batches, batch_size=batch_size)
