from facenet_pytorch import MTCNN, extract_face
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class ProcessDataset:
    LOAD_IMAGE_SIZE = (250, 250)
    IMAGE_SIZE = 160

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device, image_size=IMAGE_SIZE)

    @classmethod
    def read_imgs_from_dir(cls, path: str) -> list:
        files = os.listdir(path)

        imgs = []
        for file in files:
            file_path = os.path.join(path, file)
            img = Image.open(file_path).convert('RGB').resize(cls.LOAD_IMAGE_SIZE)
            img = np.asarray(img)
            imgs += [img]

        return imgs

    @staticmethod
    def save_images(imgs: list, boxes: list, dst_path: str):
        os.makedirs(dst_path, exist_ok=True)

        for i, (img, box) in enumerate(zip(imgs, boxes)):
            if box is None or len(box) == 0:
                continue
            extract_face(img, box[0], save_path=os.path.join(dst_path, f'{i}.jpg'))

    @staticmethod
    def crop_images(imgs: list, boxes: list):
        faces = []
        for idx, (img, box) in enumerate(zip(imgs, boxes)):
            if box is None or not len(box):
                continue
            y1, x1, y2, x2 = box[0].astype(int).clip(min=0)
            faces += [img[y1:y2][x1:x2]]
        return faces

    @classmethod
    def process(cls, dir_src: str, dir_dst: str):
        for class_folder in tqdm(os.listdir(dir_src)):
            class_folder_path = os.path.join(dir_src, class_folder)
            # print(class_folder_path)

            imgs = cls.read_imgs_from_dir(class_folder_path)

            boxes, probs, points = cls.mtcnn.detect(imgs, landmarks=True)

            cls.save_images(imgs, boxes, os.path.join(dir_dst, class_folder))



if __name__ == "__main__":

    ProcessDataset.process(
        '/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test',
        f'/home/artem/projects/face_to_vk/VGG-Face2/data/vggface2_test/test_processed_{ProcessDataset.IMAGE_SIZE}'
    )
