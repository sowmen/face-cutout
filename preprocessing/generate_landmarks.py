import os
from functools import partial
from multiprocessing.pool import Pool

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm


import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from preprocessing.utils import get_original_video_paths

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np

detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")



def save_landmarks(ori_id, root_dir):
    ori_id = ori_id[:-4]
    ori_dir = os.path.join(root_dir, "face_crops", ori_id)
    if(not os.path.exists(ori_dir)):
        return
        
    landmark_dir = os.path.join(root_dir, "landmarks", ori_id)
    os.makedirs(landmark_dir, exist_ok=True)
    
    ori_crops = os.listdir(ori_dir)
    
    for crop in ori_crops :
        if crop.endswith('.txt'):
            continue

        landmark_id = crop[:-4]
        ori_path = os.path.join(ori_dir, crop)
        landmark_path = os.path.join(landmark_dir, landmark_id)

        try:
            image_ori = cv2.imread(ori_path, cv2.IMREAD_COLOR)[...,::-1]
            frame_img = Image.fromarray(image_ori)
            batch_boxes, conf, landmarks = detector.detect(frame_img, landmarks=True)
            if landmarks is not None:
                landmarks = np.around(landmarks[0]).astype(np.int16)
                np.save(landmark_path, landmarks)
        except Exception as e:
            print("Error extracting landmark on %s, %s" % (ori_path, e))
            pass

    f = open(os.path.join(landmark_dir,'done.txt'), 'w')
    f.close()
        

ROOT_DIR = 'dfdc_train_root'
def main():
    ids = get_original_video_paths(ROOT_DIR, basename=True)
    print(f'Before : {len(ids)}')

    _temp = []
    for _ in ids:
        video_id = _[:-4]
        if(os.path.exists(os.path.join(ROOT_DIR, 'landmarks', video_id, 'done.txt'))):
            continue
        else:
            _temp.append(_)
    
    print(f'After : {len(_temp)}')
    ids = _temp
    
    os.makedirs(os.path.join(ROOT_DIR, "landmarks"), exist_ok=True)
    with Pool(processes=os.cpu_count()-4) as p:
        with tqdm(total=len(ids)) as pbar:
            func = partial(save_landmarks, root_dir=ROOT_DIR)
            for v in p.imap_unordered(func, ids):
                pbar.update()
                
                
if __name__ == "__main__":
    main()