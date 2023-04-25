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
Allsegment = np.zeros([hemorrhage_diagnosis_array.shape[0],
                       new_size[0], new_size[1]], dtype=np.uint8)

train_path = Path('train')
image_path = train_path / 'image'
label_path = train_path / 'label'
if not train_path.exists():
    train_path.mkdir()
    image_path.mkdir()
    label_path.mkdir()

counterI = 0
for sNo in tqdm(range(0 + 49, numSubj + 49)):
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

        # Saving the segmentation for a given slice
        segment_path = Path(datasetDirSubj, 'brain', str(
            sliceNos[sliceI]) + '_HGE_Seg.jpg')
        if os.path.exists(str(segment_path)):
            img = Image.open(segment_path).convert('L')
            x = img.resize(new_size)
            # Because of the resize the image has some values that are not 0
            # or 255, so make them 0 or 255
            array_from_x = np.array(x)
            # print(np.unique(array_from_x))
            array_from_x = np.where(array_from_x > 128, 255, 0)
            # print(np.unique(array_from_x)) only 0 and 255 values - correct
            x = Image.fromarray(array_from_x.astype(np.uint8))
            print(x.getcolors())  # everything looks fine so idk what's wrong
            x.save(os.path.join(label_path, (str(counterI) + '.png')))
        else:
            x = np.zeros([new_size[0], new_size[1]], dtype=np.uint8)
            x = Image.fromarray(x)
            x.save(os.path.join(label_path, (str(counterI) + '.png')))
        Allsegment[counterI] = x

        counterI = counterI + 1
