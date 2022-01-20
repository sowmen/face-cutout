import json
import os
from os import cpu_count
from tqdm  import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from utils import get_original_video_paths
from face_detector import FacenetDetector, VideoDataset


def process_videos(videos, root_dir):
    detector = FacenetDetector()
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, 
                        shuffle=False, 
                        num_workers=cpu_count() - 4, # > 0 error in windows 
                        batch_size=1, 
                        collate_fn=lambda x: x
            )
    
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        if(video == None or indices == None or frames == None):
            continue
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
        for j, frames in enumerate(batches):
            result.update(
               {
                   int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))
               } 
            )
        
        id = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(root_dir, "boxes")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)
        
 
ROOT_DIR = "dfdc_train_root"
def main():
    originals = get_original_video_paths(ROOT_DIR)
    _ = []
    for path in originals:
        video_id = Path(path).name[:-4]
        if(os.path.exists(os.path.join(ROOT_DIR, 'boxes', video_id + ".json"))):
            continue
        else:
            _.append(path)
    originals = _
    print("Processing %s videos" % (len(originals)))
    process_videos(originals, ROOT_DIR)
    
if __name__ == "__main__":
    main()
    
    
'''
    TODO:
        Change detector from MTCNN in face_detector.py
'''