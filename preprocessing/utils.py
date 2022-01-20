import json
import os
from glob import glob
from pathlib import Path

'''
    Get the list of original video paths
    params:
        root_dir - root directory containing all individual folders
        basename - If true return only list of filenames
                   else returns path to videos
'''
def get_original_video_paths(root_dir, basename=False):
    originals_path = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
            
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals_path.add(os.path.join(dir, original))
    originals_path = list(originals_path)
    originals_v = list(originals_v)
    print("Number of original videos :", len(originals_path))
    
    return originals_v if basename else originals_path



'''
    Get the list of original and fake video name pairs
    params:
        root_dir - root directory containing all individual folders
'''
def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
            
        for k,v in metadata.items():
            original = v.get("original", None)
            if(v["label"] == "FAKE"):
                pairs.append((original[:-4], k[:-4]))
                
    return pairs
