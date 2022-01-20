import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset


class VideoFaceDetector(ABC):
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    @property
    @abstractmethod    
    def _batch_size(self) -> int:
        pass
    
    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass
    
    
class FacenetDetector(VideoFaceDetector):
    
    def __init__(self, device="cuda:0") -> None:
        super().__init__()
        self.detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
    
    @property
    def _batch_size(self):
        return 8
    
    def _detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes]
    
    
class VideoDataset(Dataset):

    '''
        params:
            videos - List of video paths
    '''
    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    '''
        Extracts all frames and returns 
            video path
            indexes of read frames
            frames as PIL images
    '''
    def __getitem__(self, index: int):
        video = self.videos[index]
        try:
            capture = cv2.VideoCapture(video)
            frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = OrderedDict()
            for i in range(frames_num):
                capture.grab()
                success, frame = capture.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize(size=[s // 2 for s in frame.size]) # Resizes both dimension by half
                frames[i] = frame
            return video, list(frames.keys()), list(frames.values())
        except Exception as e:
            print("Frame extract error on video %s: %s" % (video, str(e)))
            return None, None, None

    def __len__(self) -> int:
        return len(self.videos)
        
