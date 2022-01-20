# Face-Cutout
Code for the paper *Towards Solving the DeepFake Problem: An Analysis on Improving DeepFake Detection Using Dynamic Face Augmentation* **Sowmen Das**, **Selim Seferbekov**, **Arup Datta**, **Md. Saiful Islam**, **Md. Ruhul Amin**; <span style="color:#2596be
">Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops</span>, 2021, pp. 3776-3785. [[*paper*]](https://openaccess.thecvf.com/content/ICCV2021W/RPRMI/html/Das_Towards_Solving_the_DeepFake_Problem_An_Analysis_on_Improving_DeepFake_ICCVW_2021_paper.html) [[*presentation*]](https://youtu.be/RuXKBWanx40?t=6545)

<p align="center">
  <img src="https://github.com/sowmen/face-cutout/blob/main/test_samples/intro.png">
</p>

*Sensory cutout* uses MTCNN face detector. </br>
*Convex-hull cutout* uses DLIB face detector.

## Files
- `face_cutout.py` containes implementation of Face Cutout
- See `face_cutout_demo.ipynb` to run face-cutout on sample images
- `libs` directory contains weights for DLIB Face detector.


## Instructions
```bash
   $ pip3 install virtualenv
   $ virtualenv -p python3 venv 
   $ source venv/bin/activate
   $ pip install -r requirements.txt
```
---------------------------------------------------------------------------------


## Training

Organize the data in the following structure. Each dataset will have the folders :
- `face_crops` containing face images cropped from the video.
- `diffs` containing difference masks for the fake videos. It is better to precalculate these masks for faster training. Otherwise, they will be determined during runtime.
- `original_landmarks` containing MTCNN landmarks for the original frames. Data is stored as `.npy` numpy format.
  
```bash
.
├── train_data
│   ├── <dataset>
│   │   ├── face_crops
│   │   │   ├── <video>
│   │   │       └── <framenumber_person>.png
│   │   │   .
│   │   │   .
│   │   ├── diffs
│   │   │   ├── <video>
│   │   │       └── <framenumber_person_diff>.png
│   │   │   .
│   │   │   .
│   │   ├── original_landmarks
│   │   │   ├── <video>
│   │           └── <framenumber_person>.npy
│   │   │   .
│   │   │   .

```

### CSV Structure

| video       | file                    | label | original | frame | fold | dataset |
| ----------- | -----------             | ------ | -------- | ----- | ---- | ------ |
| video name  | <frame_person>.png      |   0 / 1    | original video name for the corresponding fake video. For real videos, original is same as video | frame number | cross-validation fold | dfdc / celebdf/ ffpp


## Validation

Download `validation_dataset.zip` and pretrained weights from the releases. Extract the zip file. It contains 1650 images and a csv file for testing the models. </br>

Check `validate_images.ipynb` to reproduce the results.

```
Face Cutout --------------
100%|██████████| 40/40 [00:05<00:00,  8.04it/s]

AUC : 0.9659584773104254
mAP : 0.9927165331395946
LogLoss : 0.2119294066442086

Random Erase -------------
100%|██████████| 40/40 [00:05<00:00,  8.10it/s]
AUC : 0.9448474498375792
mAP : 0.988081434519885
LogLoss : 0.24963293145048693
```

### Citation
If you use this code in your research, please cite:

```bibtex
@InProceedings{Das_2021_ICCV,
    author    = {Das, Sowmen and Seferbekov, Selim and Datta, Arup and Islam, Md. Saiful and Amin, Md. Ruhul},
    title     = {Towards Solving the DeepFake Problem: An Analysis on Improving DeepFake Detection Using Dynamic Face Augmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {3776-3785}
}
```

### Acknowledgements

The original implementation was taken from Selim Seferbekov's DFDC submission. Follow this repository for further details:

- https://github.com/selimsef/dfdc_deepfake_challenge

