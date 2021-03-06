# %%
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

#Global paths / variables
path = "OID/Dataset/train/Vehicle registration plate/"
train_df = pd.DataFrame(columns=['FilePath','Filename','XMin', 'XMax', 'YMin', 'YMax', 'ClassName','X','Y','Width','Height'])

#Gets the annotations and images for the specific dataset
#Not needed if you already downloaded images

# %%
def setup():
    base_path = 'OID/csv_folder'
    annotations_bbox_fname = 'train-annotations-bbox.csv'
    class_descriptions_fname = 'class-descriptions-boxable.csv'
    
    #annotations
    annotations_bbox = pd.read_csv(os.path.join(base_path, annotations_bbox_fname))
    # print(annotations_bbox.head())

    #Specific class label
    class_descriptions = pd.read_csv(os.path.join(base_path, class_descriptions_fname),header=None)
    class_descriptions.columns = ["name","class"]
    # print(class_descriptions.head())
    
    #Get the Plate images
    plate_pd = class_descriptions[class_descriptions['class']=='Vehicle registration plate']
    label_plate = plate_pd['name'].values[0]

    plate_bbox = annotations_bbox[annotations_bbox['LabelName']==label_plate]
    return plate_bbox

# %%
#Get the plates images to store in pandas
def get_plates(plate_bbox):
    
    # #Amount of images in dataset
    # plate_img_id = plate_bbox['ImageID']

    # #Amount of unique images
    # plate_img_id = np.unique(plate_img_id)
    # print('There are %d plates in the dataset' %(len(plate_img_id)))

    #Find unique images
    files = os.listdir(path)
    train_plate = []
    for f in files:
        f = f[0:-4]
        train_plate.append(f)

    #gets the unique images
    rand_plates = list(set(train_plate))
    
    print('There are %d plates in the dataset' %(len(rand_plates)))

    global train_df

    temp_train = []
    for t in rand_plates:
        temp_train.append("/" + t + '.jpg')
    
    for x in temp_train:
        train_path = path + x
        isFile = os.path.isfile(train_path)
        #checks to see if path exists
        if(isFile):
            train_df = train_df.append({'FilePath': train_path,'Filename': x[1:-4],'ClassName':"0"},ignore_index=True)

    #labels folder append info to that
    lfiles = os.listdir(os.path.join(path,"Label"))
    
    for a in lfiles:
        if(train_df['Filename'].str.contains(a[0:-4]).any()):
            index = train_df.index[train_df["Filename"] == a[0:-4]].tolist()
            image = cv2.imread(train_df['FilePath'][index[0]])
            image_height, image_width, channel = image.shape
            with open(os.path.join(os.path.join(path,"Label"),a)) as labeled_file:
                csv_reader = csv.reader(labeled_file, delimiter=" ")
                temp_xmin = [] #left x
                temp_ymax = [] #top Y
                temp_xmax = [] #right x
                temp_ymin = [] #bottom y
                temp_x = []; temp_y = []; temp_width = []; temp_height = []
                for row in csv_reader:
                    absolute_width = abs(float(row[3]) - float(row[5]))
                    absolute_height = abs(float(row[4]) - float(row[6]))
                    absolute_x = float(row[3]) + absolute_width/2 
                    absolute_y = float(row[4]) + absolute_height/2

                    x = absolute_x / image_width
                    y = absolute_y / image_height
                    width = absolute_width / image_width
                    height = absolute_height / image_height
                    
                    temp_xmin.append(row[3]) #left x
                    temp_ymin.append(row[4]) #top Y
                    temp_xmax.append(row[5]) #right x
                    temp_ymax.append(row[6]) #bottom y

                    temp_x.append(str(x))
                    temp_y.append(str(y))
                    temp_width.append(str(width))
                    temp_height.append(str(height))
                #get index of file
                train_df.at[index[0], 'XMin'] = temp_xmin
                train_df.at[index[0], 'YMax'] = temp_ymax   
                train_df.at[index[0], 'XMax'] = temp_xmax 
                train_df.at[index[0], 'YMin'] = temp_ymin
                
                train_df.at[index[0], 'X'] = temp_x
                train_df.at[index[0], 'Y'] = temp_y   
                train_df.at[index[0], 'Width'] = temp_width 
                train_df.at[index[0], 'Height'] = temp_height
    return train_df

    #Yolo Formatting
    # 0 0.716797 0.395833 0.216406 0.147222
    # 0 0.687109 0.379167 0.255469 0.158333

# %%
#New Normalized Labels
def new_labels(train_df):
    file_name = train_df['Filename']
    for i in range(len(file_name)):
        new_file = file_name[i] + ".txt"
        with open(os.path.join(path,new_file),"w+") as text_file:
            for j in range(len(train_df['X'][i])):
                line = "0 " + train_df['X'][i][j] + " " \
                            + train_df['Y'][i][j] + " " \
                            + train_df['Width'][i][j] + " " \
                            + train_df['Height'][i][j] + "\n"
                text_file.write(line)

# %%
#Train and Test File
def train_test(train_df):
    #C:\Users\bhogv\ObjectDetection\OID\Dataset\train\Vehicle registration plate\0a0a00b2fbe89a47.jpg
    file_name = train_df['Filename']
    # Create and/or truncate train.txt and test.txt
    with open('test.txt', 'w') as test:
        pass
    with open('train.txt', 'w') as train:
        pass

    for i in range(len(train_df)):
        file_path = "C:/Users/bhogv/ObjectDetection/OID/Dataset/train/Vehicle registration plate/" + file_name[i] + ".jpg"
        if i < 100:
            # save to test
            with open('test.txt', 'a') as test:
                test.write(file_path + "\n")
        else:
            # save to train
            with open('train.txt', 'a') as train:
                train.write(file_path + "\n")

# %%
#plot bounding box
def plot_box(file_img,train_df):
    #train_df = pd.DataFrame(columns=['FilePath','Filename','XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
    gray_image = cv2.imread(file_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(gray_image,cv2.COLOR_BGR2RGB)
    height, width, channel = image.shape
    print(f"Image: {image.shape}")
    cv2.resize(image,(128,128))
    for index in range(len(train_df["X"][0])):
        # xmin = int(float(train_df['XMin'][0][index]))
        # xmax = int(float(train_df['XMax'][0][index]))
        # ymin = int(float(train_df['YMin'][0][index]))
        # ymax = int(float(train_df['YMax'][0][index]))

        nor_x = float(train_df['X'][0][index])
        nor_y = float(train_df['Y'][0][index])
        nor_width = float(train_df['Width'][0][index])
        nor_height = float(train_df['Height'][0][index])
        class_name = train_df['ClassName'][0]

        print(nor_x,nor_y,nor_width,nor_height)

        abs_x = nor_x * width
        abs_y = nor_y * height
        abs_width = nor_width * width
        abs_height = nor_height * height

        x1 = int(abs_x - abs_width/2) # row[3] = xmin
        y1 = int(abs_y - abs_height/2) # row[4] = ymax
        x2 = int(abs_width + x1) # row[5] = xmax
        y2 = int(abs_height + y1) # row[6] = ymin

        # print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
        print(f"Coordinates: {x1,y2}, {x2,y1}")

        cv2.rectangle(image, (x1,y2), (x2,y1), (0,0,255), 1)
        # cv2.rectangle(image, (int(abs_x),int(abs_y)), (x2,y1), (0,0,255), 1)
        # cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Plate", (x1,y1-5), font, 1, (0,0,0), 2)
    plt.title((train_df['Filename'][0]) + ".jpg")
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# %%
def main():
    #gets all files for Vehicle Registration Plate
    plate_bbox = setup()
    train_df = get_plates(plate_bbox)
    # print(train_df)
    new_labels(train_df)
    train_test(train_df)

    # #Get one image from train_df
    # file_img = train_df['FilePath'][0] 
    # plot_box(file_img,train_df)

if __name__ == "__main__":
    main()


# %%
