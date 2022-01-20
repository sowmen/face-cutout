import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from tqdm import tqdm

from preprocessing.utils import get_original_with_fakes

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_paths(vid, label, root_dir):
    ori_vid, fake_vid = vid
    ori_dir = os.path.join(root_dir, "face_crops", ori_vid)
    fake_dir = os.path.join(root_dir, "face_crops", fake_vid)
    data = []
    if(label == 0):
        try:
            if(os.path.exists(ori_dir)):
                for image_id in os.listdir(ori_dir):
                    if(image_id.endswith('txt')):
                        continue
                    ori_img_path = os.path.join(ori_dir, image_id)
                    data.append([ori_img_path, label, ori_vid])
        except:
            pass
    else:
        try:
            if(os.path.exists(fake_dir)):
                for image_id in os.listdir(fake_dir):
                    if(image_id.endswith('txt')):
                        continue
                    fake_img_path = os.path.join(fake_dir, image_id)
                    data.append([fake_img_path, label, ori_vid])
        except:
            pass

    return data

ROOT_DIR = 'dfdc_train_root'
def main(n_splits):
    ori_fakes = get_original_with_fakes(ROOT_DIR)
    
    sz = 50 // n_splits
    folds = []
    for fold in range(n_splits):
        folds.append(list(range(sz * fold, sz * fold + sz if fold < n_splits - 1 else 50)))
    print(folds)
      
    video_fold = {}
    
    for d in os.listdir(ROOT_DIR):
        if "dfdc" in d:
            part = int(d.split("_")[-1])
            for f in os.listdir(os.path.join(ROOT_DIR, d)):
                if "metadata.json" in f:
                    with open(os.path.join(ROOT_DIR, d, "metadata.json")) as metadata_json:
                        metadata = json.load(metadata_json)

                    for k, v in metadata.items():
                        fold = None
                        for i, fold_dirs in enumerate(folds):
                            if part in fold_dirs:
                                fold = i
                                break
                        assert fold is not None
                        video_id = k[:-4]
                        video_fold[video_id] = fold
                        
    data = []
    ori_ori = set([(ori, ori) for ori, fake in ori_fakes])
    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(ori_ori)) as pbar:
            func = partial(get_paths, label=0, root_dir=ROOT_DIR)
            for v in p.imap_unordered(func, ori_ori):
                pbar.update()
                data.extend(v)
        with tqdm(total=len(ori_fakes)) as pbar:
            func = partial(get_paths, label=1, root_dir=ROOT_DIR)
            for v in p.imap_unordered(func, ori_fakes):
                pbar.update()
                data.extend(v)
                
    fold_data = []
    for img_path, label, ori_vid in tqdm(data):
        path = Path(img_path)
        video = path.parent.name
        file = path.name
        assert video_fold[video] == video_fold[ori_vid], "original video and fake have leak  {} {}".format(ori_vid,
                                                                                                           video)
        fold_data.append([video, file, label, ori_vid, int(file.split("_")[0]), video_fold[video]])
    random.shuffle(fold_data)
    pd.DataFrame(fold_data, columns=["video", "file", "label", "original", "frame", "fold"]).to_csv('full_folds.csv', index=False)
    

if __name__ == "__main__":
    main(n_splits=10)
