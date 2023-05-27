import pandas as pd
import shutil
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import stat
from IPython.display import Image, display
from ultralytics import YOLO
from icrawler.builtin import GoogleImageCrawler
import random
import imgaug.augmenters as iaa
import glob
import logging
import timm
from fastai.vision.all import *
from fastai.vision.all import vision_learner, get_image_files
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger("ultralytics").setLevel(logging.ERROR)

#Images folder contains images of different items.
#In the df_classes.csv file: <item attribute name>, <item article> = df_class.
#In the train.csv file: rows with file name and df_class_id.
#The test.csv file contains lines of image names, for which you need to provide a df_class identifier for this items image.

#Copy all files from ./images/images to ./images_processing
original_folder = './images'
target_folder = './images_processing/'
train_folder = './train'
test_folder = './test/'
dataset = './dataset/'
additional_dataset = './dataset1/'

c = pd.read_csv("./df_classes.csv", index_col=False)
c = c.drop(columns=[c.columns[0]])
c = c.rename(columns = {'df_id': 'df_class_id'})

#If the class is empty, then write to image = ''
tr = pd.read_csv("./train.csv", index_col=False)
for _, i in enumerate(set(c.df_class_id)):
    if len(tr[tr.df_class_id == _]) == 0:
        tr.loc[tr.shape[0], 'df_class_id'] = _
        tr.loc[tr.shape[0] - 1, 'image'] = ''
tr.df_class_id = tr.df_class_id.astype(int)

df = tr.join(c.set_index('df_class_id'), on='df_class_id')
df = df.reset_index(drop=True)
df = df.sort_values(by = 'df_class_id')
df = df.reset_index(drop=True)
df['item_article'] = df['attribute'] + " " + df['article'] + " " + "item"

te = pd.read_csv("./test.csv", index_col=False)

if not os.path.exists(target_folder):
    os.makedirs(target_folder)
else:
    shutil.rmtree(target_folder, ignore_errors=True)

for root, dirs, files in os.walk(original_folder):
    for file in files:
        path_file = os.path.join(root, file)
        shutil.copy(path_file, f'{target_folder}/')

# h_m = []
# w_m = []
# for i in df.image:
#     try:
#         img_size = cv2.imread(f'{target_folder}/{i}', cv2.IMREAD_UNCHANGED)
#         h, w = img_size.shape[:2]
#         h_m.append(h)
#         w_m.append(w)
#     except Exception as e:
#         #print(i, e)
#         pass

#h_m = int(np.mean(h_m))
h_m = 512

#w_m = int(np.mean(w_m))
w_m = 512

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
else:
    shutil.rmtree(train_folder, ignore_errors=True)
    
# Copy to train folder by df_class_id
for c in set(df.df_class_id):
    if not os.path.exists(f'{train_folder}/{c}'):
        os.makedirs(f'{train_folder}/{c}')
    else:
        shutil.rmtree(f'{train_folder}/{c}', ignore_errors=True)
        
for c in set(df.df_class_id):
    for i in df.image[df.df_class_id == c]:
        if i:
            shutil.copy(f'{target_folder}/{i}', f'{train_folder}/{c}')
            
h_to_train = h_m
w_to_train = w_m

train_subfolders = [f.path for f in os.scandir(train_folder) if f.is_dir()]

for s in train_subfolders:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    for j in i:
        try:
            img_j = cv2.imread(f'{s}/{j}', cv2.IMREAD_UNCHANGED)
            h, w = img_j.shape[:2]
            if h < (h_m // 2) or w < (w_m // 2):
                os.remove(f'{s}/{j}')
            elif h > h_to_train or w > w_to_train:
                img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_AREA)
                os.remove(f'{s}/{j}')
                cv2.imwrite(f'{s}/{j}', img_j_)
            elif h < h_to_train or w < w_to_train:
                img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_CUBIC)
                os.remove(f'{s}/{j}')
                cv2.imwrite(f'{s}/{j}', img_j_)
            elif h == h_to_train and w == w_to_train:
                pass
        except Exception as e:  # Can't open the image
            print(f'{s}/{j}', e)
            os.remove(f'{s}/{j}')
            
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
else:
    shutil.rmtree(test_folder, ignore_errors=True)
    
for i in te.image:
    shutil.copy(f'{target_folder}{i}', test_folder)
    
test_files = [f for f in listdir(test_folder) if isfile(join(test_folder, f))]

for s in test_files:
    try:
        img_j = cv2.imread(f'{test_folder}{s}', cv2.IMREAD_UNCHANGED)
        h, w = img_j.shape[:2]
        if h > h_to_train or w > w_to_train:
            img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_AREA)
            os.remove(f'{test_folder}{s}')
            cv2.imwrite(f'{test_folder}{s}', img_j_)
        elif h < h_to_train or w < w_to_train:
            img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_CUBIC)
            os.remove(f'{test_folder}{s}')
            cv2.imwrite(f'{test_folder}{s}', img_j_)
        elif h == h_to_train and w == w_to_train:
            pass
    except Exception as e:
        #os.remove(f'{test_folder}{s}')  # Images don't open, make a random class for such images
        print(f'{test_folder}{s}', e)

# Custom object detection 
##################################Binary_object detection for next custom model
# train_subfolders = [f.path for f in os.scandir(train_folder) if f.is_dir()]
# images = []
# for s in train_subfolders:
#     i = [f for f in listdir(s) if isfile(join(s, f))]
#     for j in i:
#         try:
#             img_j = cv2.imread(f'{s}/{j}', cv2.IMREAD_UNCHANGED)
#             h, w, c = img_j.shape
#         except Exception as e:
#             images.append(f'{s}/{j}')
#             #display(Image(filename=f'{s}/{j}'))
#             #print(f'{s}/{j}', e)
            

# class_folder = './class/'

# if not os.path.exists(class_folder):
#     os.makedirs(class_folder)
# else:
#     shutil.rmtree(class_folder, ignore_errors=True)
    
# for i in images:
#     shutil.copy(i, class_folder)


###################################################### Remove images with tables (class = 0)
#https://docs.ultralytics.com/modes/predict/#boxes
# import os
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/***')

# !pip install ultralytics
# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="***")
# project = rf.workspace("***").project("***")
# dataset = project.version(1).download("yolov8")
# model = YOLO("yolov8n.pt")

# from ultralytics import YOLO

# model.train(data="/content/***", epochs=200)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# success = model.export(format="onnx")  # export the model to ONNX format

############################## ONNX model
# model = YOLO("best.onnx")
# path_test = r'./images_processing/'

# i = [f for f in listdir(path_test) if isfile(join(path_test, f))]

# for file in i:
#     try:
#         inputs = f'{path_test}{file}'
#         results = model(inputs)  # List of results objects
#         boxes = results[0].boxes
#         box = boxes[0]  # returns one box
#     except Exception as e:
#         print(inputs, e)

# Continue with GPU (copy train folder)
# import onnxruntime as rt
# rt.get_device()

train_subfolders = [f.path for f in os.scandir(train_folder) if f.is_dir()]
model = YOLO("best.onnx")  # Faster model inference

for s in train_subfolders:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    for j in i:
        try:
            inputs = f'{s}/{j}'
            results = model(inputs)
            boxes = results[0].boxes
            box = boxes[0]

            if int(boxes.cls[0]) == 0:  # If it's not an item
                #display(Image(filename=inputs))
                #print(int(boxes.cls[0]), boxes.conf[0])
                os.remove(inputs)
        except Exception as e:
            #print(inputs, e)
            pass
        
# Data balance
files_per_class = {}
for i in set(df.df_class_id):
    class_name = f'{train_folder}/{i}'
    if i not in files_per_class:
        files_per_class[i] = len([name for name in os.listdir(class_name) if os.path.isfile(os.path.join(class_name, name))])

value_per_class = int(np.percentile(list(files_per_class.values()), 90))

df_classes = pd.DataFrame(data=set(df.df_class_id), columns=['df_class_id'])
df_classes['item_article'] = np.nan
df_classes['value'] = np.nan

for i, j in files_per_class.items():
    if j < value_per_class:
        df_classes.loc[i, 'item_article'] = list(set(df[df.df_class_id == i].item_article))[0]
        df_classes.loc[i, 'value'] = value_per_class - j

df_classes = df_classes.dropna()
df_classes.value = df_classes.value.astype(int)

# Creating custom dataset
if not os.path.exists(dataset):
    os.makedirs(dataset)
else:
    shutil.rmtree(dataset, ignore_errors=True)
    
filters = dict(
type="photo",
license='commercial,modify')
    
#https://buildmedia.readthedocs.org/media/pdf/icrawler/latest/icrawler.pdf
for name, quantity, path in zip(df_classes.item_article, df_classes.value, df_classes.df_class_id):
    google_crawler = GoogleImageCrawler(feeder_threads=1, parser_threads=1, downloader_threads=1,
                                        storage={'root_dir': f'{dataset}/{path}'},
                                        log_level=logging.CRITICAL)
    google_crawler.crawl(min_size=(512,512), max_size=None, keyword=name, max_num=quantity)
    
#Copy to other folder and resize to 512*512 images
dataset_subfolders = [f.path for f in os.scandir(additional_dataset) if f.is_dir()]

for s in dataset_subfolders:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    for j in i:
        try:
            img_j = cv2.imread(f'{s}/{j}', cv2.IMREAD_UNCHANGED)
            h, w = img_j.shape[:2]
            if h > h_to_train or w > w_to_train:
                img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_AREA)
                os.remove(f'{s}/{j}')
                cv2.imwrite(f'{s}/{j}', img_j_)
            elif h < h_to_train or w < w_to_train:
                img_j_ = cv2.resize(img_j, (h_to_train, w_to_train), interpolation = cv2.INTER_CUBIC)
                os.remove(f'{s}/{j}')
                cv2.imwrite(f'{s}/{j}', img_j_)
            elif h == h_to_train and w == w_to_train:
                pass
        except Exception as e:  # Can't open the image
            print(f'{s}/{j}', e)
            os.remove(f'{s}/{j}')
            
model = YOLO("best.onnx")  # Faster model inference

for s in dataset_subfolders:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    for j in i:
        try:
            inputs = f'{s}/{j}'
            results = model(inputs)
            boxes = results[0].boxes
            box = boxes[0]

            if int(boxes.cls[0]) == 0 or (int(boxes.cls[0]) == 1 and boxes.conf[0] < 0.2):
                #display(Image(filename=inputs))
                #print(int(boxes.cls[0]), boxes.conf[0])
                os.remove(inputs)
        except Exception as e:
            print(inputs, e)
            #os.remove(inputs)
    
#Copy from dataset to train_folder by df_class_id
for root, dirs, files in os.walk(additional_dataset):
    for file in files:
        path_file = os.path.join(root, file)
        shutil.copy(path_file, f"{train_folder}/{root.split('/')[len(root.split('/')) - 1]}")
        
#Delete images from items where images > value_per_df_class
train_files = [f.path for f in os.scandir(train_folder) if f.is_dir()]
for s in train_files:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    if len(i) > value_per_class:
        print(s, i)
        #for file in i[:len(i) - value_per_class]:
        #    os.remove(f'{s}/{file}')
        for file in random.sample(i, len(i) - value_per_class):
            os.remove(f'{s}/{file}')

#arch = 'convnext_large'
arch = 'convnext_xlarge_in22k'
#arch = 'convnext_huge_in22k'

trn_path = './train_example'
files = get_image_files(trn_path)

#function to help us minimise the workflow
def train(arch, item, batch, epochs=5):
    dls = ImageDataLoaders.from_folder(trn_path, seed=42, valid_pct=0.2, bs=17, item_tfms=item, batch_tfms=batch)
    metrics=[accuracy]
    learn = vision_learner(dls, arch, metrics=metrics).to_fp32() #loss function is cross entropy ,default
    learn.fine_tune(epochs, 0.01)
    return learn

learn = train(arch, item=Resize((512,512), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros), epochs=10, batch=aug_transforms(size=(512,512)))
learn.show_results()

# Dataset of images augmentation 
images = []
augmentation = iaa.Sequential([
    # 1. Flip
    iaa.Fliplr(0.1),
    iaa.Flipud(0.1),
    # 2. Affine
    #iaa.Affine(#translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
               #rotate=(-10, 10),
#),
    # 3. Multiply
    iaa.Multiply((1, 1.2)),
    # 4. Linearcontrast
    iaa.LinearContrast((0.9, 1.1)),
    # Perform methods below only sometimes
    iaa.Sometimes(1,
        # 5. GaussianBlur
        iaa.GaussianBlur((0.0, 2.0))
        )
])

train_files = [f.path for f in os.scandir(train_folder) if f.is_dir()]
for s in train_files:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    if len(i) < value_per_class:
        for _, file in enumerate(range(value_per_class - len(i))):
            for img_path in i:
                img = cv2.imread(f'{s}/{img_path}', cv2.IMREAD_UNCHANGED)
                images.append(img)
                augmented_images = augmentation(images=images)
            for j in augmented_images:
                cv2.imwrite(f'{s}/{_}_{img_path}', j)
                #print(f'{s}/{_}_{img_path}')

train_files = [f.path for f in os.scandir(train_folder) if f.is_dir()]
for s in train_files:
    i = [f for f in listdir(s) if isfile(join(s, f))]
    if len(i) != value_per_class:
        print(s, len(i))
