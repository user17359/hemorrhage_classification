# This code loads the greyscale images of the brain-window
# for each subject in Patient_CT folder.
# The code resizes the images and saves them to one folder (train\image).

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

numSubj = 82
new_size = (512, 512)
currentDir = Path(os.getcwd())
datasetDir = str(Path(currentDir, 'Patients_CT'))

# Reading labels
hemorrhage_diagnosis_df = pd.read_csv(
    Path(currentDir, 'hemorrhage_diagnosis.csv')
)
hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values

# reading images
AllCTscans = np.zeros([hemorrhage_diagnosis_array.shape[0],  # number of columns (?)
                       new_size[0], new_size[1]], dtype=np.uint8)

train_path = Path('train')
image_path = train_path / 'image'

if not train_path.exists():
    train_path.mkdir()
    image_path.mkdir()

counterI = 0
for sNo in tqdm(range(0+49, numSubj+49)):
    datasetDirSubj = Path(datasetDir, "{0:0=3d}".format(sNo))

    idx = hemorrhage_diagnosis_array[:, 0] == sNo
    sliceNos = hemorrhage_diagnosis_array[idx, 1]
    NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]
    for sliceI in range(0, sliceNos.size):
        img_path = Path(datasetDirSubj, 'brain',
                        str(sliceNos[sliceI]) + '.jpg')
        img = Image.open(img_path)
        x = img.resize(new_size)
        AllCTscans[counterI] = x
        x.save(os.path.join(image_path, (str(counterI) + '.png')))
