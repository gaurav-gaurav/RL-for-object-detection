import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import pandas as pd
import numpy as np 
from PIL import Image
import cv2
import os
train_image_path = 'VOCdevkit/VOC2007/JPEGImages/'
image_list = os.listdir(train_image_path)
for i in range(20):
    img_name = image_list[i]
    img_path = train_image_path + img_name
    img = Image.open(img_path)
    img.thumbnail((128,128), Image.ANTIALIAS)
    print(img.size)





# # class Reshape(nn.Module):
# #     def __init__(self, *args):
# #         super(Reshape, self).__init__()


# #     def forward(self, x):
# #         return x.view(x.size(0),-1)

# class ConvNN(nn.Module):
# 	"""docstring for ConvNN"""
# 	def __init__(self):
# 		super(ConvNN, self).__init__()
# 		self.conv1 = nn.Conv2d(3,8,4,2)
# 		self.conv2 = nn.Conv2d(8,16,3,2)
# 		self.conv3 = nn.Conv2d(16,32,3,2)
# 		self.conv4 = nn.Conv2d(32,64,3,2)
# 		self.conv5 = nn.Conv2d(64,128,3,1)
# 		self.conv6 = nn.Conv2d(128,256,3,1)
# 		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		
# 		self.fc1 = nn.Linear(256,100)
# 		self.fc2  = nn.Linear(100,3)

# 	def forward(self, x):
# 		model1  = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.avgpool)
# 		model2 = nn.Sequential(self.fc1, self.fc2)
# 		x =model1(x)
# 		x = x.view(x.size(0), -1)
# 		x = model2(x)
		

# 		return x
# model = ConvNN().to('cuda:0')
# summary(model, (3,128,128))

# # df= pd.read_csv('labels.csv')
# # class_name = list(df['class'])
# # print(len(class_name))
# # row = [i for i, j in enumerate(class_name) if j == 'hand' or j=='head' or j=='foot']
# # print(len(row))
# # df.drop(row, axis = 0, inplace=True)
# # class_name = list(df['class'])
# # print(len(class_name))

# def letterbox_image(img, inp_dim):
#     '''resize image with unchanged aspect ratio using padding'''
#     img = np.array(img)
#     img_w, img_h = img.shape[1], img.shape[0]
#     w, h = inp_dim

#     new_w = int(img_w * min(w/img_w, h/img_h))
#     new_h = int(img_h * min(w/img_w, h/img_h))
#     resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
#     canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

#     canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
#     canvas = Image.fromarray(canvas.astype('uint8'), 'RGB')
#     return canvas,new_w,new_h

# from gluoncv import model_zoo, data, utils
# from matplotlib import pyplot as plt

# net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

# im_fname = 'dog.jpg'

# x, img = data.transforms.presets.yolo.load_test(im_fname, mean=(0,0,0), std = (0,0,0))
# # print(type(x), type(img))
# # x = np.array(Image.open(im_fname))
# # print(type(x))
# # print('Shape of pre-processed image:', x.shape)
# class_IDs, scores, bounding_boxs = net(x)

# ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
#                          class_IDs[0], class_names=net.classes)
# # plt.show()