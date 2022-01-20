## Data Preprocessing

The preprocessing steps are same for all datasets. The files provided cannot be used as is. I have only provided them as I have used. For individual purposes fix teh data paths and output directories.

1. `detect_original_faces.py` -> Detect original faces from only the original videos. This will save `.json` containing bounding box locations for the faces in each frame of aeach video.
2. `extract_crops.py` -> Extract the face regions using the bounding boxes and save them as `.png`.
3. `generate_landmarks.py` -> Identify the face landmarks using MTCNN face detector and save them as `.npy` files. This step is optional as landmark identification can be done during training while runtime. But precalculating is preferred for faster training.
4. `generate_ssim_diffs.py` -> Generate and save the SSIM difference masks used for facecutout. This is also optional, but speeds up runtime calculations.
5. `generate_data_folds.py` -> Create train/validation folds by seperating source faces. This ensures that there is no data leak for testing.


