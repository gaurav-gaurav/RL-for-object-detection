import numpy as np
import numpy
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
from datetime import datetime
import time  

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
from policy import policy_network
from PPO import memory_buffer, ppo

from util import *
from darknet import Darknet
from preprocess import inp_to_image
from yolo_detect import Detector

####################____________________Arguments_Parser____________________####################

parser = argparse.ArgumentParser('RL_for_image_correction')
parser.add_argument('--eval', type = bool, default = 0)

parser.add_argument('--dataset', type = str, default = 'voc')
parser.add_argument('--path_of_dataset', type = str, default = 'data/voc')
parser.add_argument('--image_dim', default= (128, 128))

parser.add_argument('--epochs', type = int, default = 400)
parser.add_argument('--steps', type = int, default = 2000)
parser.add_argument('--train_itr', type = int, default = 20)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--seed', type = int, default =123)
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--lr', type = float, default= 1e-3)
parser.add_argument('--lambda', type = float, default = 0.95)
parser.add_argument('--reduce_lr', type  = bool, default =  False)
parser.add_argument('--value_clip', type = bool, default = False)
parser.add_argument('--pretrain', type = bool, default = False)
parser.add_argument('--value_coef', type = float, default = 1.0)

parser.add_argument('--attention', type = bool, default = False)
parser.add_argument('--action_type', type  =  str, default = 'discrete')
parser.add_argument('--action_size', type = int, default= 1)
parser.add_argument('--noise_scale', type = str, default = 'full') ## clip
parser.add_argument('--action_class', type = int, default = 20) # 14
parser.add_argument('--min_act', type = float, default = 0.1) # 0.5
parser.add_argument('--max_act', type = float, default = 2)  # 1.8

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

parser.add_argument('--resume', type = bool , default= False)
parser.add_argument('--resume_dir', type=str, default='', help="path to resume file")
parser.add_argument('--save_dir', type = str, default = 'log/dis_full')
parser.add_argument('--train_image_path', type = str, default = 'VOCdevkit/VOC2007/')
parser.add_argument('--labels', type =  str, default = 'labels.csv')

parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

####################____________________Data_Loader_and_Model_Initilaizer____________________####################

img_label = get_label('labels.csv')
train_image_path = args.train_image_path + str('JPEGImages/')
annotations = args.train_image_path + str('Annotations/')

num_images = len(os.listdir(train_image_path)) 
np.random.seed(args.seed)
eps = np.finfo(np.float32).eps.item()
classes = load_classes('data/coco.names')
Colors = np.random.randint(0,255, size = (len(classes), 3), dtype = 'uint8')

policy = policy_network(args)
memory = memory_buffer(args)
PPO = ppo(args)
detector  = Detector(args)

optimizer = optim.Adam(policy.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.99)

####################____________________Set_Cuda_and_Resume_Training____________________####################

CUDA = torch.cuda.is_available()
if CUDA:
    print('Success: Cuda available')
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', torch.cuda.memory_allocated(0)/1024**3,'GB')
    print('Cached:', torch.cuda.memory_reserved(0)/1024**3, 'GB')
    policy.to(args.device)

if args.resume:
    print('Loading model to resume training...')
    PATH = args.resume
    checkpoint = torch.load(PATH, map_location = 'cpu')
    policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    policy.train()

else:
    print('#'*5,'_'*5,'No model to load','_'*5,'#'*5)
print(args)

####################____________________Save_Model_and_Reward____________________####################

def save_model(path):
    torch.save({'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        path)

def read_csv(path):
    reward_summary = open(path + '/Reward_data.csv', 'r')
    reward_data = np.genfromtxt(reward_summary)
    reward_summary.close()
    try:
        return [i for i in reward_data] 
    except TypeError:
        return [reward_data]

def write_csv(path, Reward_list):
    reward_summary1 = open(path + '/Reward_data.csv', 'w')
    np.savetxt(reward_summary1, Reward_list)
    reward_summary1.close()

####################____________________Main_Function____________________####################

def main():
    image_list = os.listdir(train_image_path)
    writer = SummaryWriter(args.save_dir)

    for epoch in tqdm(range(args.epochs)): 
        image_list = shuffle_arr(image_list)
        Reward = []
        img_path_list = []

        step = 0
        while True:
            img_name =  image_list[step]
            img_path = train_image_path + img_name
            img = Image.open(img_path)
            img_array = np.array(img)

####################____Preprocess_Image_and_Add_Noise____####################

            orig_img, w,h = letterbox_image(img, args.image_dim) ## returns PIL class object 
            orig_img_array = np.array(orig_img)
            gt_img_array = copy.deepcopy(orig_img_array)
            original_detect = detector.detector(orig_img_array)        ##[batch_idx, Xmin, Ymin, Xmax, Ymax, object_score, class_score, class_index] 

            if args.action_type=='discrete':
                noisey_img, noise_scale = synthetic_change_dis(args, orig_img, flag = 1)
            if args.action_type == 'continuous':
                noisey_img, noise_scale = synthetic_change_cont(args, orig_img, flag =1)
            noisey_img_array = np.array(noisey_img)
            noisey_detect = detector.detector(noisey_img_array) ##(128,128,3)

####################____Refine_Image____####################

            transform = torchvision.transforms.ToTensor()
            state = transform(noisey_img_array) # (3,128,128)
            state = state.unsqueeze(0).to(args.device)
            with torch.no_grad():
                action_, action, action_prob, value = policy(state)

            refined_img = change_brightness(noisey_img, action_)
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

####################____Plot_Results____####################

            if args.show_result:
                show_result(orig_img_array, noisey_img_array, refined_img_array)
            if args.show_bbox:
                plot_image(orig_img_array, original_detect, classes, Colors,'original_image')
                plot_image(noisey_img_array, noisey_detect, classes, Colors, 'noisey_image')
                plot_image(refined_img_array, refined_detect, classes, Colors, 'refined_image')
                plot_GT_image(gt_img_array, resized_gnd_truth_arr, classes, Colors, 'Ground_truth')

####################____Compute_Reward____####################

            if len(original_detect) and len(noisey_detect) and len(refined_detect) >0:
                original_pred_array = pred_array(original_detect, classes)
                noisey_pred_arryay = pred_array(noisey_detect, classes)
                refined_pred_array = pred_array(refined_detect, classes)
            else:
                continue

            if (type(original_pred_array) != numpy.ndarray) or (type(noisey_pred_arryay) != numpy.ndarray) or (type(refined_pred_array) != numpy.ndarray):
                continue 
            print (resized_gnd_truth_arr.shape)
            score_orig = get_score(args, resized_gnd_truth_arr, original_pred_array, args.iou_threshold, eps)
            score_noise = get_score(args, resized_gnd_truth_arr, noisey_pred_arryay, args.iou_threshold, eps)
            score_refine = get_score(args, resized_gnd_truth_arr, refined_pred_array, args.iou_threshold, eps)
            
            BETA = 2*score_refine - score_orig - score_noise
            if BETA>= 0.01:
                reward  = 1
            else:
                reward = -1
            Reward.append(reward)

####################____Update_Memory_Buffer____####################

            memory.update_buffer(step, noise_scale, action, action_prob, reward, value)
            img_path_list.append(img_path)
            step +=1
            if step == args.steps:
                break
            
####################____Train_and_Print_Summary____####################

        img_path_list = np.array(img_path_list)

        for i in range(args.train_itr):
            batch_idx = np.random.randint(0,args.steps,args.batch_size)
            noise_scale, old_action, old_action_prob, old_reward, old_value = memory.fetch_data(batch_idx)
            index = list(batch_idx)
            image_path = img_path_list[index[:]]
            image_batch = get_image_batch(args, image_path, noise_scale)

            b_action = old_action.reshape(-1)
            b_log_prob = old_action_prob.reshape(-1)
            b_image = image_batch
            b_reward= old_reward.reshape(-1)
            b_value = old_value.reshape(-1)

            with torch.no_grad():
                advantage = PPO.get_advantage(b_reward, b_value)

            b_advantage = advantage.reshape(-1)
            b_return = torch.add(b_value, b_advantage)

            *_, new_log_prob, new_value = policy(b_image, b_action)
            cost = PPO.get_ppo_loss(b_log_prob, new_log_prob, b_advantage, b_reward, new_value)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        mean_reward = round(sum(Reward)/len(Reward),4)
        writer.add_scalar('Reward:', mean_reward, epoch)
        
        Reward_list = read_csv(args.save_dir)
        Reward_list.extend([mean_reward])
        Reward_list = np.array(Reward_list)
        write_csv(args.save_dir, Reward_list)
        save_path = osp.join(args.save_dir, str(epoch) + '.pt')
        save_model(save_path)
        
if __name__ == '__main__':
    main()
