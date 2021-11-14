import numpy as np
import cv2
import pandas as pd
import os

from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, PadIfNeeded
from transforms import IsotropicResize


class Test_Dataset(Dataset):
    def __init__(self, df, root_dir, size=224):

        super().__init__()
        self.df = df
        self.root_dir = root_dir
        self.normalize = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

        self.data = self.df.values
        np.random.shuffle(self.data)

        self.transforms = Compose(
            [
                IsotropicResize(
                    max_side=size,
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_CUBIC,
                ),
                PadIfNeeded(
                    min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        filename, label = self.data[index]
        path = os.path.join(self.root_dir, filename)

        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error: {path} {e}")

        data = self.transforms(image=image)
        image = data["image"]
        image = img_to_tensor(image, self.normalize)

        return {"image": image, "label": label}
