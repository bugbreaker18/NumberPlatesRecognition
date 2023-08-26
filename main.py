import os
import numpy as np
import cv2
import uuid
import time
import pandas as pd
import xmltodict
import glob
import xml.etree.ElementTree as ET
import random as rnd
import splitfolders
import easyocr
import PIL
import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from PIL import Image
from tqdm.auto import tqdm
from numba import cuda
from timeit import default_timer as timer
import trainyolo
import torch

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib import patches as mpatches


#print("COOL")

dataset = {
            "file":[],
            "width":[],
            "height":[],
            "xmin":[],
            "ymin":[],
            "xmax":[],
            "ymax":[]
           }
img_names=[]
annotations=[]
for dirname, _, filenames in os.walk("carlicenseplate"):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:]==("png" or "jpg"):
            img_names.append(filename)
        elif os.path.join(dirname, filename)[-3:]=="xml":
            annotations.append(filename)

#print(f"${img_names[:10]}")

path_annotations="carlicenseplate/annotations/*.xml"

for item in glob.glob(path_annotations):
    tree = ET.parse(item)

    for elem in tree.iter():
        if 'filename' in elem.tag:
            filename=elem.text
        elif 'width' in elem.tag:
            width=int(elem.text)
        elif 'height' in elem.tag:
            height=int(elem.text)
        elif 'xmin' in elem.tag:
            xmin=int(elem.text)
        elif 'ymin' in elem.tag:
            ymin=int(elem.text)
        elif 'xmax' in elem.tag:
            xmax=int(elem.text)
        elif 'ymax' in elem.tag:
            ymax=int(elem.text)

            dataset['file'].append(filename)
            dataset['width'].append(width)
            dataset['height'].append(height)
            dataset['xmin'].append(xmin)
            dataset['ymin'].append(ymin)
            dataset['xmax'].append(xmax)
            dataset['ymax'].append(ymax)

classes = ['license']
#print("COOL")

df=pd.DataFrame(dataset)
#print(df)
#df.info()



def print_random_images(photos: list, n: int = 5, seed=None) -> None:
    if n > 10:
        n=10

    if seed:
        rnd.seed(seed)

    random_photos = rnd.sample(photos, n)

    for image_path in random_photos:

        with Image.open(image_path) as fd:
            fig, ax = plt.subplots()
            ax.imshow(fd)
            ax.axis(False)

            for i, file in enumerate(df.file):
                if file in image_path:
                    x1,y1,x2,y2=list(df.iloc[i, -4:])

                    mpatch=mpatches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1, edgecolor='b',facecolor="none",lw=2,)
                    ax.add_patch(mpatch)
                    rx, ry = mpatch.get_xy()
                    ax.annotate('licence', (rx, ry-2), color='blue', weight='bold', fontsize=12, ha='left', va='baseline')

photos_path = "carlicenseplate/images/*.png"
photos_list = glob.glob(photos_path)
print(len(photos_list))
print_random_images(photos_list)



from pathlib import Path

# Create the output directory for annotations
annotations_path = Path("working/annotations")
annotations_path.mkdir(parents=True, exist_ok=True)

x_pos = []
y_pos = []
frame_width = []
frame_height = []

previous_filename = None

for i, row in enumerate(df.iloc):
    current_filename = str(row.file[:-4])

    width, height, xmin, ymin, xmax, ymax = list(df.iloc[i][-6:])

    x = (xmin + xmax) / 2 / width
    y = (ymin + ymax) / 2 / height
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height

    x_pos.append(x)
    y_pos.append(y)
    frame_width.append(width)
    frame_height.append(height)

    txt = f'0 {x} {y} {width} {height}\n'

    # Determine save type
    save_type = 'a+' if current_filename == previous_filename else 'w'

    with open(annotations_path / f"{current_filename}.txt", save_type) as f:
        f.write(txt)

    previous_filename = current_filename

df['x_pos'] = x_pos
df['y_pos'] = y_pos
df['frame_width'] = frame_width
df['frame_height'] = frame_height

#print(df)

input_folder = Path("/home/bugbreaker18/PycharmProjects/carlicenseplate/carlicenseplate")
output_folder = Path("/home/bugbreaker18/PycharmProjects/carlicenseplate/yolo/")
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.8, 0.2),
    group_prefix=None
)
#print("Moving files finished.")




def walk_through_dir(dir_path: Path) -> None:
    """Prints dir_path content"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directiories and {len(filenames)} files in '{dirpath}' folder ")


walk_through_dir(input_folder)
print()
walk_through_dir(output_folder)

# TRY YOLOV8 instead of YOLOV5
print("Input folder exists:", input_folder.exists())
print("Output folder exists:", output_folder.exists())


trainyolo.trainedyolo()
