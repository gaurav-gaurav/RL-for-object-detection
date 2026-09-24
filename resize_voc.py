import PIL
import os
import cv2
from PIL import Image
import numpy as np
import csv
import pandas as pd 

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img = np.array(img)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim

    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    canvas = Image.fromarray(canvas.astype('uint8'), 'RGB')
    return canvas,new_w,new_h

def getResizedBB(arr,w,h,width,height,reso):
    if height <= width:
        xmin=arr[0]*w/width
        xmax=arr[2]*w/width
        ymin=arr[1]*h/height+(reso-h)/2
        ymax=arr[3]*h/height+(reso-h)/2
        
    elif height > width:
        xmin=arr[0]*w/width+(reso-w)/2
        xmax=arr[2]*w/width+(reso-w)/2
        ymin=arr[1]*h/height
        ymax=arr[3]*h/height
        
    return [xmin,ymin,xmax,ymax]

def get_label(data):
    df= pd.read_csv(data)
    class_name = list(df['class'])
    row = [i for i, j in enumerate(class_name) if j == 'hand' or j=='head' or j =='foot']
    df.drop(row, axis = 0, inplace=True)
    return df

data_path = 'VOCdevkit/VOC2007/JPEGImages/'
image_list = os.listdir(data_path)
img_label = get_label('labels.csv')

with open('resized_labels.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['filename', 'xmin', 'ymin', 'xmax','ymax','width', 'height', 'class'])
	for img_name in image_list:
		img_path = data_path + img_name
		img = Image.open(img_path)
		img_array = np.array(img)
		resized_image, w,h = letterbox_image(img, (128,128))
		resized_image.save('VOCdevkit/VOC2007/resized_img/' + img_name)

		ground_truth_df = img_label[img_label['filename'] == img_name]
		ground_truth_arr = []

		for i in range(len(ground_truth_df)):
			ground_truth_arr.append(ground_truth_df.iloc[i][:])
		ground_truth_arr = np.array(ground_truth_arr)
		resized_gnd_truth_arr = np.copy(ground_truth_arr)
		for i in range(len(ground_truth_arr)):
			arr = ground_truth_arr[i][1:5]
			t = getResizedBB(arr,w,h,img_array.shape[1], img_array.shape[0], 128)
			resized_gnd_truth_arr[i][1:5]= np.array(t)
		writer.writerows(resized_gnd_truth_arr)
