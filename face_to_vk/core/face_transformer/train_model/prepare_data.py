import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import os
from multiprocessing import Process, Queue

from typing import Optional


# random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataLoader:

    TARGET_SIZE = (160, 160)
    TORCH_TRANSFORM = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.classes = {class_: os.listdir(os.path.join(self.dataset_path, class_))
                        for class_ in os.listdir(self.dataset_path)}

        self.p: Optional[Process] = None
        self.queue: Optional[Queue] = None

    def create_triplet_batch(self, batch_size: int = 16):
        """
        Create batch with anchor, positive and negative image.

        :param batch_size: batch size, default 16
        :returns: 3 tensors with shape [batch_size, 3, height, width]
        """

        batch_dict = {0: [], 1: [], 2: []}

        for batch_num in range(batch_size):
            class1, class2 = random.sample(self.classes.keys(), 2)

            image1, image2 = random.sample(self.classes[class1], 2)
            image3 = random.choice(self.classes[class2])

            image1 = Image.open(os.path.join(self.dataset_path, class1, image1))
            image2 = Image.open(os.path.join(self.dataset_path, class1, image2))
            image3 = Image.open(os.path.join(self.dataset_path, class2, image3))

            # apply transform for each image
            for i, image in enumerate((image1, image2, image3)):
                image = self.TORCH_TRANSFORM(image)
                image = image[None, :]
                batch_dict[i] += [image]

        return torch.cat(batch_dict[0]).to(device), \
               torch.cat(batch_dict[1]).to(device), \
               torch.cat(batch_dict[2]).to(device)

    def _put_batch_to_queue(self, queue_size: int = 1, batch_size: int = 16):
        while True:
            if self.queue.qsize() < 5:
                self.queue.put(self.create_triplet_batch(batch_size))

    def start_async_reader(self, queue_size: int = 1, batch_size: int = 16):
        """
        Start async image loader.

        :param queue_size:
        :param batch_size: batch size
        """
        if self.p is not None:
            return

        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
        self.queue = Queue()
        self.p = Process(target=self._put_batch_to_queue, args=(queue_size, batch_size,))
        self.p.daemon = True
        self.p.start()
        print("Success started async data loader")

    def stop_async_reader(self):
        """
        Stop async loader.
        """
        if not self.p:
            return

        self.p.kill()
        self.queue.close()
        self.p = None
        self.queue = None
        print("Success stopped async data loader")


if __name__ == "__main__":
    data_loader = CustomDataLoader('/VGG-Face2/data/vggface2_test/test_processed_160')

    import time
    start_t = time.time()
    data_loader.start_async_reader(queue_size=5)

    while time.time() - start_t < 5:
        print(data_loader.queue.get()[0].shape)
        time.sleep(0.1)

    data_loader.stop_async_reader()

