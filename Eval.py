import numpy as np 
import argparse
import os
import random
from random import shuffle
from tqdm import tqdm
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt 
import cv2
import os.path as osp
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.autograd import Variable as V
from torchsummary import summary
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from utils import *
from Policy import policy_network
from PPO import memory_buffer, ppo
from resnet_policy import resnet18

from util import *
from darknet import Darknet
from preprocess import inp_to_image
from yolo_detect import Detector

####################____________________Arguments_Parser____________________####################

parser = argparse.ArgumentParser('RL_for_image_correction')
parser.add_argument('--eval', type = bool, default = 1)

parser.add_argument('--dataset', type = str, default = 'voc')
parser.add_argument('--path_of_dataset', type = str, default = 'data/voc')
parser.add_argument('--cnn_arch', type = str, default = 'pretrain')
parser.add_argument('--pretrain', type = str, default = 'resnet')
parser.add_argument('--image_dim', type  =int, default= (416, 416))

parser.add_argument('--epoch', type = int, default = 1)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--lr', type = float, default= 1e-4)
parser.add_argument('--lambda', type = float, default = 0.95)
parser.add_argument('--number_of_layers', type = int, default = 2)
parser.add_argument('--number_of_neurons',  type = int, default = 32)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--trajectory _length', type = int, default = 50)
parser.add_argument('--feature_length', type = int, default =512)
parser.add_argument('--reduce_lr', type  = bool, default =  False)

parser.add_argument('--attention', type = bool, default = False)
parser.add_argument('--action_type', type  =  str, default = 'discrete')
parser.add_argument('--action_size', type = int, default= 1)
parser.add_argument('--action_class', type = int, default =40)
parser.add_argument('--noise_scale', type = str, default = 'full')
parser.add_argument('--min_act', type = float, default = 0.5)
parser.add_argument('--max_act', type = float, default = 1.8)

parser.add_argument('--noise_type',  type = int, default = 1, 
                    help = '1 for brightness, 2 for color, 3 for contrast 4 for sharpness 5 for all')

parser.add_argument('--show_result', type = bool, default = False)
parser.add_argument('--show_bbox', type = bool , default = False)
parser.add_argument('--noise_correction_summary', type = bool, default = False)

parser.add_argument('--confidence',type=float,default=0.5,
                    help='Confidence Threshold for Object Detection(YOLO)')
parser.add_argument('--nms_thresh',type=float,default=0.3,
                    help='Non Maximal Suppression Threshold for YOLO')
parser.add_argument('--iou_threshold',type=float,default=0.5,
                    help='Threshold for IOU to determine if object is detected or not')
parser.add_argument('--alpha',type=float,default=0.1,
                    help='IOU Weight for reward --> r=alpha*(iou)+(1-alpha)*F1')

parser.add_argument('--eval_dir', type=str, default='log/cont_full_no_attn/5000.pt', help="path to resume file")
parser.add_argument('--save_dir', type = str, default = 'log/dis_resnet_full_no_attn')
parser.add_argument('--train_image_path', type = str, default = 'VOCdevkit/VOC2007/')
parser.add_argument('--labels', type =  str, default = 'labels.csv')

parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

####################____________________Data_Loader_and_Model_Initilaizer____________________####################

img_label = get_label('labels.csv')
train_image_path = args.train_image_path + str('JPEGImages/')
annotations = args.train_image_path + str('Annotations/')
num_images = len(os.listdir(train_image_path)) 

policy = policy_network(args)
optimizer = optim.Adam(policy.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.99)

eps = np.finfo(np.float32).eps.item()
classes = load_classes('data/coco.names')
Colors = np.random.randint(0,255, size = (len(classes), 3), dtype = 'uint8')
detector  = Detector(args)

if args.noise_scale == 'full':
    args.min_act = 0
    args.max_act = 2
####################____________________Set_Cuda_and_Resume_Training____________________####################

CUDA = torch.cuda.is_available()
if CUDA:
    print('Success: Cuda available')
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', torch.cuda.memory_allocated(0)/1024**3,'GB')
    print('Cached:', torch.cuda.memory_reserved(0)/1024**3, 'GB')
    policy.to(args.device)

if args.eval:
    print('Loading model to Evaluate.....')
    PATH = args.eval_dir
    checkpoint = torch.load(PATH, map_location = 'cpu')
    policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    policy.eval()

else:
    print('#'*5,'_'*5,'No model to load','_'*5,'#'*5)


####################____________________Main_Function____________________####################

def main():
    image_list = os.listdir(train_image_path)
    writer = SummaryWriter(args.save_dir)
    reward_epoch = []
    Temp_reward_list = []
    itr = 0
    TP_improvement_count = 0 
    TP_degrade_count = 0
    for epoch in range(args.epoch): 
        image_list = shuffle_arr(image_list)
        reward_arr = []

        for episodes in tqdm(range(num_images)):
            itr += 1
            img_name =  image_list[episodes]
            img_path = train_image_path + img_name
            img = Image.open(img_path)
            img_array = np.array(img)
            
####################____Preprocess_Image_and_Add_Noise____####################

            orig_img, *_ = letterbox_image(img, args.image_dim)          ## image array for learning
            orig_img_array = np.array(orig_img)
            gt_img_array = copy.deepcopy(orig_img_array)
            # original_detect = detector.detector(orig_img_array)        ##[batch_idx, Xmin, Ymin, Xmax, Ymax, object_score, class_score, class_index] 

            if args.action_type=='discrete':
                noisey_img_, noise_factor = synthetic_change_dis(args, img, flag = 1)
            if args.action_type == 'continuous':
                noisey_img_, noise_factor = synthetic_change_cont(args, img, flag =1)

            noisey_img, *_ = letterbox_image(noisey_img_, args.image_dim)
            noisey_img_array = np.array(noisey_img)
            noisey_detect = detector.detector(noisey_img_array) 

####################____Refine_Image____####################

            state = preprocess_img(args, noisey_img_array)
            action, action_prob, value = policy.get_action(policy, state)

            refined_img = change_brightness(noisey_img_, action)
            refined_img, w, h = letterbox_image(refined_img, args.image_dim)
            refined_img_array = np.array(refined_img)
            refined_detect = detector.detector(refined_img_array) 
 
####################____Get_Ground_Truth_Bounding_Box____####################

            ground_truth_df = img_label[img_label['filename'] == img_name]
            ground_truth_arr = []

            for i in range(len(ground_truth_df)):
                ground_truth_arr.append(ground_truth_df.iloc[i][1:])
            ground_truth_arr = np.array(ground_truth_arr)
            resized_gnd_truth_arr = np.copy(ground_truth_arr)

            for i in range(len(ground_truth_arr)):
                arr = ground_truth_arr[i][:4]
                t = getResizedBB(arr,w,h,img_array.shape[1], img_array.shape[0], args.image_dim[0])
                resized_gnd_truth_arr[i][:4]= np.array(t)

#############################
            pred_refine = np.array(refined_detect.cpu())
            pred_refine_arr = []
            for i in range(len(pred_refine)):
                arr = list(pred_refine[i][1:5])

                arr.append(classes[int(pred_refine[i][-1])])
                arr = np.array(arr)
                pred_refine_arr.append(arr)
            pred_refine_arr = np.array(pred_refine_arr)
            # print(pred_refine_arr)
#############################
            pred_noise = np.array(noisey_detect.cpu())
            pred_noise_arr = []
            for i in range(len(pred_noise)):
                arr = list(pred_noise[i][1:5])

                arr.append(classes[int(pred_noise[i][-1])])
                arr = np.array(arr)
                pred_noise_arr.append(arr)
            pred_noise_arr = np.array(pred_noise_arr)
            # print(pred_refine_arr)
####################____Plot_Results____####################

            if args.show_result:
                show_result(orig_img_array, noisey_img_array, refined_img_array)
            
            if args.show_bbox:
                plot_image(orig_img_array, original_detect, classes, Colors,'original_image')
                plot_image(noisey_img_array, noisey_detect, classes, Colors, 'noisey_image')
                plot_image(refined_img_array, refined_detect, classes, Colors, 'refined_image')
                plot_GT_image(gt_img_array, resized_gnd_truth_arr, classes, Colors, 'Ground_truth')

####################____Compute_TP_Score____####################

            TP_refined, *_ = get_F1(resized_gnd_truth_arr, pred_refine_arr, args.iou_threshold)
            TP_noise, *_ = get_F1(resized_gnd_truth_arr, pred_noise_arr, args.iou_threshold)
            # print('noise_action', noise_factor)
            # print('correction_action', action)
            if TP_refined > TP_noise:
                TP_improvement_count += 1
                print('TP_improvement_count:',TP_improvement_count)
            if TP_refined < TP_noise:
                TP_degrade_count +=1
                print('TP_degrade_count:',TP_degrade_count)
                
            # print(TP_refined, TP_noise)
            # recall = TP/(TP+FN+eps)
            # precision = TP/(TP+FP+eps)

            # F1 = 2*recall*precision/(precision+recall+eps)
            # if len(iou)>0:
            #     iou_reward = np.mean(iou)
            # else:
            #     iou_reward = 0

            # reward = args.alpha*(iou_reward) + (1-args.alpha)*F1
            # reward_arr.append(reward)  

        
if __name__ == '__main__':
    main()
