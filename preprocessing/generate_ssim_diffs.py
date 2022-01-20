import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import partial
from multiprocessing.pool import Pool

from preprocessing.utils import get_original_with_fakes
from skimage.measure import compare_ssim

from tqdm import tqdm

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import numpy as np

cache = {}


def save_diffs(pair, root_dir):
    ori_id, fake_id = pair
    ori_dir = os.path.join(root_dir, "face_crops", ori_id)
    fake_dir = os.path.join(root_dir, "face_crops", fake_id)
    diff_dir = os.path.join(root_dir, "diffs", fake_id)
    os.makedirs(diff_dir, exist_ok=True)
    
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            diff_image_id = "{}_{}_diff.png".format(frame, actor)
            ori_path = os.path.join(ori_dir, image_id)
            fake_path = os.path.join(fake_dir, image_id)
            diff_path = os.path.join(diff_dir, diff_image_id)
            if os.path.exists(ori_path) and os.path.exists(fake_path):
                img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
                try:
                    d, a = compare_ssim(img1, img2, multichannel=True, full=True)
                    a = 1 - a
                    diff = (a * 255).astype(np.uint8)
                    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(diff_path, diff)
                except:
                    pass

    f = open(os.path.join(diff_dir,'done.txt'), 'w')
    f.close()
                

ROOT_DIR = 'dfdc_train_root'
def main():
    pairs = get_original_with_fakes(ROOT_DIR)
    
    _temp = []
    for ori_id, fake_id in tqdm(pairs):
        if(os.path.exists(os.path.join(ROOT_DIR, 'diffs', fake_id, 'done.txt'))):
            continue
        else:
            _temp.append((ori_id, fake_id))
    print(f"Total : {len(pairs)}")
    print(f"After : {len(_temp)}")
    pairs = _temp
    
    os.makedirs(os.path.join(ROOT_DIR, "diffs"), exist_ok=True)
    
    with Pool(processes=os.cpu_count() - 1) as p:
        with tqdm(total=len(pairs)) as pbar:
            func = partial(save_diffs, root_dir=ROOT_DIR)
            for v in p.imap_unordered(func, pairs):
                pbar.update()
                
                
if __name__ == "__main__":
    main()

    
    