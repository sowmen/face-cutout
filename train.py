import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import gc
import json
import wandb
import pandas as pd

from torch.backends import cudnn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cudnn.benchmark = True

from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HorizontalFlip,
    FancyPCA,
    HueSaturationValue,
    OneOf,
    ToGray,
    ShiftScaleRotate,
    ImageCompression,
    PadIfNeeded,
    GaussNoise,
    GaussianBlur,
)
from transforms import IsotropicResize

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from dataset import FaceDataset
from sklearn import metrics

from utils import EarlyStopping, AverageMeter, get_dataframe


OUTPUT_DIR = "weights"
device = "cuda"
config_defaults = {
    "epochs": 5,
    "train_batch_size": 8,
    "valid_batch_size": 8,
    "optimizer": "radam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0005,
    "schedule_patience": 3,
    "schedule_factor": 0.25,
    "rand_seed": 777,
    "cutout_fill": 0,
}

VAL_FOLD = 5

def train(name, df):

    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    print("Starting -->", run)
    
    wandb.init(project="dfdc", config=config_defaults, name=run)
    config = wandb.config


    model = timm.create_model("tf_efficientnet_b4_ns", pretrained=False, num_classes=1)
    # model.load_state_dict(
    #     torch.load(
    #         "weights/Combined_hardcore_min_loss,tf_efficientnet_b4_ns_fold_5_run_1.h5"
    #     )
    # )
    model.to(device)
    # model = DataParallel(model).to(device)

    if config.optimizer == "radam":
        optimizer = torch_optimizer.RAdam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.schedule_patience,
        threshold=0.001,
        mode="min",
        factor=config.schedule_factor,
    )
    criterion = nn.BCEWithLogitsLoss()
    es = EarlyStopping(patience=10, mode="min")

    data_train = FaceDataset(
        mode="train",
        df=df,
        val_fold=VAL_FOLD,
        cutout_fill=config.cutout_fill,
        cutout=True,
        random_erase=False,
        equal_sample=False,
        transforms=create_train_transforms(size=224),
    )
    train_data_loader = DataLoader(data_train, batch_size=config.train_batch_size, num_workers=8, shuffle=True, drop_last=True)

    data_val = FaceDataset(
        mode="val",
        df=df,
        val_fold=VAL_FOLD,
        cutout=False,
        equal_sample=False,
        transforms=create_val_transforms(size=224),
    )
    val_data_loader = DataLoader(data_val, batch_size=config.valid_batch_size, num_workers=8, shuffle=False, drop_last=True)



    for epoch in range(config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(model, train_data_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, val_data_loader, criterion, epoch)

        scheduler.step(valid_metrics["valid_loss"])

        print(f"TRAIN_AUC = {train_metrics['train_auc']}, TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_AUC = {valid_metrics['valid_auc']}, VALID_LOSS = {valid_metrics['valid_loss']}")


        es(
            valid_metrics["valid_loss"],
            model,
            model_path=os.path.join(OUTPUT_DIR, f"{run}.h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break


def train_epoch(model, train_data_loader, optimizer, criterion, epoch):
    model.train()

    train_loss = AverageMeter()
    correct_predictions = []
    targets = []

    idx = 1
    for batch in tqdm(train_data_loader):

        batch_images = batch["image"].to(device)
        batch_labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(batch_images)

        loss = criterion(out, batch_labels.view(-1, 1).type_as(out))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), train_data_loader.batch_size)
        batch_targets = (batch_labels.view(-1, 1).cpu() >= 0.5) * 1
        batch_preds = torch.sigmoid(out).detach().cpu()
        

        targets.append(batch_targets)
        correct_predictions.append(batch_preds)


        if idx % 1000 == 0:
            with torch.no_grad():
                temp_t = np.vstack((targets)).ravel()
                temp_correct_preds = np.vstack((correct_predictions)).ravel()

                train_auc = metrics.roc_auc_score(temp_t, temp_correct_preds)
                train_f1_05 = metrics.f1_score(temp_t, (temp_correct_preds >= 0.5) * 1)
                train_acc_05 = metrics.accuracy_score(
                    temp_t, (temp_correct_preds >= 0.5) * 1
                )
                train_balanced_acc_05 = metrics.balanced_accuracy_score(
                    temp_t, (temp_correct_preds >= 0.5) * 1
                )
                train_ap = metrics.average_precision_score(temp_t, temp_correct_preds)
                train_log_loss = metrics.log_loss(
                    temp_t, expand_prediction(temp_correct_preds)
                )

                train_metrics = {
                    "b_train_loss": train_loss.avg,
                    "b_train_auc": train_auc,
                    "b_train_f1_05": train_f1_05,
                    "b_train_acc_05": train_acc_05,
                    "b_train_balanced_acc_05": train_balanced_acc_05,
                    "b_train_batch": idx,
                    "b_train_ap": train_ap,
                    "b_train_log_loss": train_log_loss,
                }
                wandb.log(train_metrics)
        idx += 1

    with torch.no_grad():
        targets = np.vstack((targets)).ravel()
        correct_predictions = np.vstack((correct_predictions)).ravel()

        train_auc = metrics.roc_auc_score(targets, correct_predictions)
        train_f1_05 = metrics.f1_score(targets, (correct_predictions >= 0.5) * 1)
        train_acc_05 = metrics.accuracy_score(targets, (correct_predictions >= 0.5) * 1)
        train_balanced_acc_05 = metrics.balanced_accuracy_score(
            targets, (correct_predictions >= 0.5) * 1
        )
        train_ap = metrics.average_precision_score(targets, correct_predictions)
        train_log_loss = metrics.log_loss(
            targets, expand_prediction(correct_predictions)
        )

    train_metrics = {
        "train_loss": train_loss.avg,
        "train_auc": train_auc,
        "train_f1_05": train_f1_05,
        "train_acc_05": train_acc_05,
        "train_balanced_acc_05": train_balanced_acc_05,
        "train_ap": train_ap,
        "train_log_loss": train_log_loss,
        "epoch": epoch,
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, val_data_loader, criterion, epoch):
    model.eval()

    valid_loss = AverageMeter()
    correct_predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader):
            batch_images = batch["image"].to(device).float()
            batch_labels = batch["label"].to(device).float()

            out = model(batch_images)
            loss = criterion(out, batch_labels.view(-1, 1).type_as(out))

            valid_loss.update(loss.item(), val_data_loader.batch_size)

            batch_targets = (batch_labels.view(-1, 1).cpu() >= 0.5) * 1
            batch_preds = torch.sigmoid(out).detach().cpu()
            

            targets.append(batch_targets)
            correct_predictions.append(batch_preds)
            

    # Logging
    targets = np.vstack((targets)).ravel()
    correct_predictions = np.vstack((correct_predictions)).ravel()

    valid_auc = metrics.roc_auc_score(targets, correct_predictions)
    valid_f1_05 = metrics.f1_score(targets, (correct_predictions >= 0.5) * 1)
    valid_acc_05 = metrics.accuracy_score(targets, (correct_predictions >= 0.5) * 1)
    valid_balanced_acc_05 = metrics.balanced_accuracy_score(
        targets, (correct_predictions >= 0.5) * 1
    )
    valid_ap = metrics.average_precision_score(targets, correct_predictions)
    valid_log_loss = metrics.log_loss(targets, expand_prediction(correct_predictions))

    valid_metrics = {
        "valid_loss": valid_loss.avg,
        "valid_auc": valid_auc,
        "valid_f1_05": valid_f1_05,
        "valid_acc_05": valid_acc_05,
        "valid_balanced_acc_05": valid_balanced_acc_05,
        "valid_ap": valid_ap,
        "valid_log_loss": valid_log_loss,
        "epoch": epoch,
    }
    wandb.log(valid_metrics)

    return valid_metrics


def expand_prediction(arr):
    arr_reshaped = arr.reshape(-1, 1)
    return np.clip(np.concatenate((1.0 - arr_reshaped, arr_reshaped), axis=1), 0.0, 1.0)


def create_train_transforms(size=224):
    return Compose(
        [
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf(
                [
                    IsotropicResize(
                        max_side=size,
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC,
                    ),
                    IsotropicResize(
                        max_side=size,
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR,
                    ),
                    IsotropicResize(
                        max_side=size,
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR,
                    ),
                ],
                p=1,
            ),
            PadIfNeeded(
                min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT
            ),
            OneOf(
                [RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7
            ),
            ToGray(p=0.2),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ]
    )


def create_val_transforms(size=224):
    return Compose(
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


if __name__ == "__main__":
    model_name = "tf_efficientnet_b4_ns"

    dfdc_df = get_dataframe("train_data/dfdc.csv", folds=10)
    ffpp_df = get_dataframe("train_data/ffpp.csv", folds=10)
    celebdf_df = get_dataframe("train_data/celebdf.csv", folds=10)

    final_df = pd.concat([dfdc_df, ffpp_df, celebdf_df])

    train(name="Combined_hardcore_" + model_name, df=final_df)