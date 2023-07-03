import os
import random
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import RandomErasing, Normalize, ToTensor
from skimage.metrics import structural_similarity as compare_ssim

import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("libs/shape_predictor_68_face_landmarks.dat")

from face_cutout import face_cutout, convex_hull_cutout, sensory_cutout

ROOT_DIR = "train_data"

class FaceDataset(Dataset):
    def __init__(
        self,
        df,
        mode,
        val_fold=5,
        cutout=True,
        cutout_fill=0,
        random_erase=False,
        equal_sample=True,  # Equalizes number of real and fake frames
        reduce_val=False,  # Reduces number of frames for faster validation
        transforms=None,  # pytorch transforms
        seed = 777,
    ):

        super().__init__()
        self.df = df
        self.mode = mode
        self.val_fold = val_fold
        self.cutout = cutout
        self.cutout_fill = cutout_fill
        self.random_erase = random_erase
        self.equal_sample = equal_sample
        self.reduce_val = reduce_val
        self.transforms = transforms
        self.seed = seed
        self.normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        if self.mode == "train":
            rows = df[df["fold"] != self.val_fold]
        else:
            rows = df[df["fold"] == self.val_fold]

        if self.equal_sample:
            rows = self._oversample(rows, self.seed)

        if self.mode == "val" and self.reduce_val:
            rows = rows[rows["frame"] % 25 == 0]

        print(
            "real {} fakes {} mode {}".format(
                len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode
            )
        )
        self.data = rows.values

        np.random.seed(self.seed)
        np.random.shuffle(self.data)

    def __getitem__(self, index: int):

        while(True):
            video, file, label, original, frame, fold, dataset = self.data[index]

            if self.mode == "train":
                label = np.clip(label, 0.01, 1 - 0.01)

            # Load image and mask
            img_path = os.path.join(ROOT_DIR, dataset, "face_crops", video, file)
            ori_path = os.path.join(ROOT_DIR, dataset, "face_crops", original, file)
            if(os.path.exists(img_path)):
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                index = random.randint(0, len(self.data) - 1)
                continue

            if(os.path.exists(ori_path)):
                ori_image = cv2.imread(ori_path, cv2.IMREAD_COLOR)
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            else:
                ori_image = None
            
            # Precalculate and save diff masks for faster loading
            if label == 1:
                mask_path = os.path.join(ROOT_DIR, dataset, "diffs", video, file[:-4]+"_diff.png")
                if os.path.exists(mask_path):
                    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    d, a = compare_ssim(ori_image, image, multichannel=True, full=True)
                    a = 1 - a
                    diff = (a * 255).astype(np.uint8)
                    mask_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                mask_image = None


            # Applying face-cutout augmentations
            if (self.mode == "train" and self.cutout):
                landmark_path = os.path.join(ROOT_DIR, dataset, "original_landmarks", original, file[:-4] + ".npy")
                if os.path.exists(landmark_path):
                    landmarks = np.load(landmark_path)
                else:
                    landmarks = None

                try:
                    # Select sensory or convex-hull randomly
                    image = face_cutout(image, ori_image, landmarks, mask_image, cutout_fill=self.cutout_fill)
                    
                    # Uncomment to use only sensory cutout
                    # image = sensory_cutout(image, ori_image, landmarks, mask_image, cutout_fill=self.cutout_fill)

                    # Uncomments to use only convex-hull cutout
                    # image = convex_hull_cutout(image, ori_image, mask_image)
                except Exception as e:
                    print(f"Augmentation Error {img_path}", e)

            if self.mode == "train" and self.random_erase:
                image = RandomErasing(p=0.5, scale=(0.02, 0.2), value="random")(image)

            # Use builtin transforms passed in
            if self.transforms is not None:
                print(type(image))
                data = self.transforms(image=image)
                image = data["image"]

            
            transNormalize = Normalize(mean=self.normalize['mean'], std=self.normalize['std'])
            transTensor = ToTensor()

            tensor_image = transTensor(image)
            tensor_image = transNormalize(tensor_image) 

            return {
                "image": tensor_image,
                "label": label,
            }

    def __len__(self) -> int:
        return len(self.data)


    """
        Equalizes count of fake and real samples
    """
    def _oversample(self, rows, seed):
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_real = real["video"].count()
        num_fakes = fakes["video"].count()
        if self.mode == "train" and num_real > num_fakes:
            fakes = fakes.sample(n=int(num_real), replace=True, random_state=seed)
        return pd.concat([real, fakes])
