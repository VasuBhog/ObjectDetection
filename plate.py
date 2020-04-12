import numpy as np
import time
import sys
import os
import random
from skimage import io
from skimage.transform import resize
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import shutil
import cv2
import csv
# import tensorflow as tf
path = "OID/Dataset/train/Vehicle registration plate"
train_df = pd.DataFrame(columns=['FilePath','Filename','XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# from keras.preprocessing.image import ImageDataGenerator

def setup():
    base_path = 'OID\csv_folder'
    images_boxable_fname = 'train-images-boxable.csv'
    annotations_bbox_fname = 'train-annotations-bbox.csv'
    class_descriptions_fname = 'class-descriptions-boxable.csv'

    #SETTING Dataframe from CSV
    images_boxable = pd.read_csv(os.path.join(base_path, images_boxable_fname))
    # print(images_boxable.head())

    annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
    # print(annotations_bbox.head())

    class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname))
    # print(class_descriptions.head())
    
    #Get the Plate images
    plate_pd = class_descriptions[class_descriptions['class']=='Vehicle registration plate']
    label_plate = plate_pd['name'].values[0]
    # print(plate_pd)
    #Find plates in annotations
    plate_bbox = annotations_bbox[annotations_bbox['LabelName']==label_plate]
    return plate_bbox

def get_box(plate_bbox):
    #train_df = pd.DataFrame(['FilePath','Filename','XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
    
    return 0


def get_plates(plate_bbox):
    #Amount of images in dataset
    print('There are %d plate in the dataset' %(len(plate_bbox)))
    plate_img_id = plate_bbox['ImageID']

    #Amount of unique images
    plate_img_id = np.unique(plate_img_id)
    print('There are %d plates in the dataset' %(len(plate_img_id)))

    # Shuffle the ids and pick the first 1000 ids
    copy_plate_id = plate_img_id.copy()
    np.random.seed(1)
    np.random.shuffle(copy_plate_id)

    n = 1000
    subplate_img_id = copy_plate_id[:n]
    # print(subplate_img_id)

    #Find images in randomn list
    files = os.listdir(path)
    train_plate = []
    for f in files:
        f = f[0:-4]
        train_plate.append(f)

    #gets the 1000 matched  images
    rand_plates = list(set(train_plate).intersection(subplate_img_id))
    return rand_plates

def plate_dataframe(rand_plates):
    temp_train = []
    for t in rand_plates:
        temp_train.append("/" + t + '.jpg')
    
    
    #copy files folder images
    for x in temp_train:
        train_path = path + x
        isFile = os.path.isfile(train_path)
        #checks to see if patht exists
        if(isFile):
            newPath = shutil.copy(train_path, "OID/Dataset/train/Vehicle registration plate/images")
            train_df = train_df.append({'FilePath': newPath,'Filename': x[1:-4],'ClassName':"Vehicle registration plate"},ignore_index=True)


    #labels folder append info to that
    lfiles = os.listdir(os.path.join(path,"Label"))
    
    for a in lfiles:
        if(train_df['Filename'].str.contains(a[0:-4]).any()):
            with open(os.path.join(os.path.join(path,"Label"),a)) as labeled_file:
                csv_reader = csv.reader(labeled_file, delimiter=" ")
                temp_xmin = [] #left x
                temp_ymax = [] #top Y
                temp_xmax = [] #right x
                temp_ymin = [] #bottom y
                for row in csv_reader:
                    temp_xmin.append(row[3]) #left x
                    temp_ymax.append(row[4]) #top Y
                    temp_xmax.append(row[5]) #right x
                    temp_ymin.append(row[6]) #bottom y
                #get index of file
                index = train_df.index[train_df["Filename"] == a[0:-4]].tolist()
                train_df.at[index[0], 'XMin'] = temp_xmin
                train_df.at[index[0], 'YMax'] = temp_ymax   
                train_df.at[index[0], 'XMax'] = temp_xmax 
                train_df.at[index[0], 'YMin'] = temp_ymin  
    print(len(train_df))
    print(train_df.head())
    return train_df

#plot bounding box
def plot_box(file_img,train_df):
    #train_df = pd.DataFrame(columns=['FilePath','Filename','XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
    gray_image = cv2.imread(file_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB)
    height, width, channel = image.shape
    print(f"Image: {image.shape}")
    cv2.resize(image,(128,128))
    for index in range(len(train_df["XMin"][0])):
        xmin = int(float(train_df['XMin'][0][index]))
        xmax = int(float(train_df['XMax'][0][index]))
        ymin = int(float(train_df['YMin'][0][index]))
        ymax = int(float(train_df['YMax'][0][index]))
        class_name = train_df['ClassName'][0]
        print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
        cv2.rectangle(image, (xmax,ymax), (xmin,ymin), (255,0,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Plate", (xmin,ymax-5), font, 1, (0,0,0), 2)
    plt.title((train_df['Filename'][0]) + ".jpg")
    plt.imshow(image)
    plt.axis("off")
    plt.show()



def train_model():
    train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        rescale=1/255
    )

    train_generator = train_datagen.flow_from_directory(
        '../data/images_cropped/quilt/open_images/',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        '../data/images_cropped/quilt/open_images/',
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary'
    )

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.losses import binary_crossentropy
    from keras.callbacks import EarlyStopping
    from keras.optimizers import RMSprop


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=binary_crossentropy,
                optimizer=RMSprop(lr=0.0005),  # half of the default lr
                metrics=['accuracy'])

def main():
    plate_bbox = setup()
    print(plate_bbox)
    get_box(plate_bbox)
    # rand_plates = get_plates(plate_bbox)
    # train_df = plate_dataframe(rand_plates)
    
    # #Get one image from train_df
    # file_img = train_df['FilePath'][0] 
    # plot_box(file_img,train_df)

if __name__ == "__main__":
    main()


#Prepare dataset format for faster rcnn code
# Save images to train and test directory
#train 0.8 and test 0.2
# train_path = os.path.join(path, 'train')
# os.mkdir(train_path)
# test_path = os.path.join(path, 'test')
# os.mkdir(test_path)


"""
images_path = "OID\Dataset\train\Vehicle registration plate\images"
all_imgs = os.listdir(images_path)
all_imgs = [f for f in all_imgs if not f.startswith('.')]
random.seed(1)
random.shuffle(all_imgs)

train_imgs = all_imgs[:800]
test_imgs = all_imgs[800:]
    
# Copy each classes' images to train directory
for j in range(len(train_imgs)):
    original_path = os.path.join(images_path, train_imgs[j])
    new_path = os.path.join(train_path, train_imgs[j])
    copyfile(original_path, new_path)

# Copy each classes' images to test directory
for j in range(len(test_imgs)):
    original_path = os.path.join(path, test_imgs[j])
    new_path = os.path.join(test_path, test_imgs[j])
    copyfile(original_path, new_path)

print('number of training images: ', len(os.listdir(train_path))) # subtract one because there is one hidden file named '.DS_Store'
print('number of test images: ', len(os.listdir(test_path)))
"""