import json
import os
from os import cpu_count
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm


'''
    Returns a tuple
        path to video if fake or real,
        path to original video bbox json
        
        If a video is real, then it returns it's own bbox json
'''
def get_video_paths(root_dir):
    paths = []
    for json_path in tqdm(glob(os.path.join(root_dir, "*/metadata.json"))):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
            
        for k,v in metadata.items():
            original = v.get("original", None)
            if original is None:
                original = k
                
            bboxes_path = os.path.join(root_dir, "boxes", original[:-4] + ".json")
            if not os.path.exists(bboxes_path):
                continue
            paths.append((os.path.join(dir, k), bboxes_path))
            
    return paths


def extract_video(param, root_dir, crops_dir):
    video, bboxes_path = param
    with open(bboxes_path, "r") as f:
        bboxes_dict = json.load(f)
        
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    done = False
    for i in range(frames_num):
        capture.grab()
        
        if i % 10 != 0: # Takes every 10th frame. Add heuristic here
            continue
        
        success, frame = capture.retrieve()
        if not success or str(i) not in bboxes_dict or bboxes_dict[str(i)] is None:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
        crops = []
        bboxes = bboxes_dict[str(i)]
        if bboxes is None:
            continue
        
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            h, w = crop.shape[:2]
            crops.append(crop)
            
        img_dir = os.path.join(root_dir, crops_dir, id)
        os.makedirs(img_dir, exist_ok=True)
        done = True

        for j, crop in enumerate(crops):
            cv2.imwrite(os.path.join(img_dir, "{}_{}.png".format(i, j)), crop)

    if(done):
        f = open(os.path.join(img_dir,'done.txt'), 'w')
        f.close()


ROOT_DIR = 'dfdc_train_root'
CROPS_DIR = 'face_crops2'
def main():
    os.makedirs(os.path.join(ROOT_DIR, CROPS_DIR), exist_ok=True)
    params = get_video_paths(ROOT_DIR) # Get video and bbox path tuples
    print("Total videos : {}".format(len(params)))

    _temp = []
    for video_path, _ in tqdm(params):
        video_id = Path(video_path).name[:-4]
        if(os.path.exists(os.path.join(ROOT_DIR, 'face_crops', video_id, 'done.txt'))):
            continue
        else:
            _temp.append((video_path, _))
    params = _temp
    print("Remaining : {}".format(len(params)))

    with Pool(processes=cpu_count()-4) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=ROOT_DIR, crops_dir=CROPS_DIR), params):
                pbar.update()
                

if __name__ == "__main__":
    main()