# face-cutout
Code for the paper "Towards Solving the DeepFake Problem: An Analysis on Improving DeepFake Detection Using Dynamic Face Augmentation"

- face_cutout.py containes implementation of Face Cutout
- face_cutout_demo.ipynb to see demo of face-cutout 
- validate_images.ipynb to see results of test run on dataset


- see requirements.txt for dependencies
- inside the extracted folder create a virtual environment and install the dependencies by running,
---------------------------------------------------------------------------------
  1. pip3 install virtualenv
  2. virtualenv -p python3 venv 
  3. source venv/bin/activate
  4. pip install -r requirements.txt
---------------------------------------------------------------------------------

- data/sample_faces contain pairs of faces with fake and original. These are used in face_cutout_demo.ipynb
- face_cutout.py containes implementation of Face Cutout
- run face_cutout_demo.ipynb to see demo of face-cutout 
- data/val_images contains 1500+ extracted faces from videos, randomly sampled from the test set as described in the paper. Due to limitation of upload size we provide the extracted faces instead of uploading the entire videos.
- run validate_images.py or validate_images.ipynb to test model on data to see AUC, mAP and LogLoss
- EfficientNet_B4 weights trained on the combined dataset are provided in weights directory
- libs directory contains weights for DLIB Face detector.