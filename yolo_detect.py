from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import torchvision

def prep_image(img, args):
    transform = torchvision.transforms.ToTensor()
    state = transform(img) # (3,128,128)
    state = state.unsqueeze(0).to(args.device)
    return state

class Detector():
    def __init__(self,args):
        self.args = args
        self.cfg_file = 'cfg/yolov3-voc.cfg'
        self.weight_file = 'Weights/yolov3-voc_final.weights'
        self.num_classes = 20
        self.confidence = args.confidence
        self.nms_thresh  =  args.nms_thresh
        self.reso = args.image_dim[0]

    def detector(self, image, batch_img_name):
        CUDA = torch.cuda.is_available()
        bbox_attrs = 5 + self.num_classes
        model = Darknet(self.cfg_file)
        model.load_weights(self.weight_file)
        model.net_info["height"] = self.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 ==0
        assert inp_dim >32
        model.to(self.args.device)
        model.eval()
        img =  image
        
        img = img.to(self.args.device)
        batch_output = model(img)
        OutPut = {}
        for i in range(len(batch_img_name)):
            img_name = batch_img_name[i]
            output_raw = torch.unsqueeze(batch_output[i,...],0)
            output = write_results(output_raw, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thresh)
            if type(output) == int:
                output = torch.tensor(np.zeros((1,8)))
            
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))

            # output[:,[1,3]] *= 128
            # output[:,[2,4]] *= 128
            OutPut[img_name] = output
        print(OutPut)
        return OutPut

        
#         output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/float(inp_dim)
     
# #            im_dim = im_dim.repeat(output.size(0), 1)
#         output[:,[1,3]] *= image.shape[1]
#         output[:,[2,4]] *= image.shape[0]
