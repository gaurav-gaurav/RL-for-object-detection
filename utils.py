import numpy as np
import pandas as pd
import random
from PIL import Image,ImageEnhance
from random import shuffle
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import copy
import torchvision

'''
shuffle is an inplace operation so they retun a none value
that is why we have creted a function and return the value
'''
def shuffle_arr(arr):
    b=arr[:]
    shuffle(b)
    return b

def change_brightness(image,brightness_factor):
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(brightness_factor)
    return image
    
def change_contrast(image,contrast_factor):
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(contrast_factor)
    return image

def change_color(image,color_factor):
    color = ImageEnhance.Color(image)
    image = color.enhance(color_factor)
    return image

def change_sharpness(image,sharpness_factor):
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(sharpness_factor)
    return image
    
    
def read(filepath,size):
    new_width = size
    new_height = size
    img = Image.open(filepath)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img

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

def synthetic_change_dis(args,img,flag, act = None):
    action_table_synth = np.linspace(args.min_act,args.max_act,args.action_class)

    if flag==1:
        if act==None:
            act = round(random.choice(action_table_synth),1)
        change_img = change_brightness(img,act)

    elif flag==2:
        if act==None:
            act = round(random.choice(action_table_synth),1)
        change_img = change_color(img,act)
        
    elif flag==3:        
        if act==None:
            act = round(random.choice(action_table_synth),1)
        change_img = change_contrast(change_img,act)

    elif flag==4: 
        if act==None:
            act = round(random.choice(action_table_synth),1)
        change_img = change_sharpness(change_img,act)

    elif flag ==5:
        if act==None:
            act1 = round(random.choice(action_table_synth),1)
        else:
            act1 = act[0]
        change_img = change_brightness(img,act1)
        
        if act==None:
            act2 = round(random.choice(action_table_synth),1)
        else:
            act2 = act[1]
        change_img = change_color(change_img,act2)
        
        if act==None:
            act3 = round(random.choice(action_table_synth),1)
        else:
            act3 =act[2]
        change_img = round(random.choice(action_table_synth),1)

        if act==None:
            act4 = round(random.choice(action_table_synth),1)
        else:
            act4 = act[3]
        change_img = change_sharpness(change_img,act4)
        
        act=[act1,act2,act3,act4]
    else:
    	raise ValueError('Incorrect flag please try from 1 to 5')

    return change_img, act

def synthetic_change_cont(args, img, flag, act = None):

    if flag==1:
        if act==None:
            act =  np.random.uniform(args.min_act, args.max_act)
        change_img = change_brightness(img,act)

    elif flag==2:
        act = np.random.uniform(args.min_act, args.max_act)
        change_img = change_color(img,act)
        
    elif flag==3:        
        if act==None:
            act =  np.random.uniform(args.min_act, args.max_act)
        change_img = change_contrast(change_img,act3)

    elif flag==4: 
        if act==None:
            act =  np.random.uniform(args.min_act, args.max_act)
        change_img = change_sharpness(change_img,act4)

    elif flag ==5:
        if act==None:
            act1 =  np.random.uniform(args.min_act, args.max_act)
        else:
            act1 = act[0]
        change_img = change_brightness(img,act1)
        
        if act==None:
            act2 =  np.random.uniform(args.min_act, args.max_act)
        else:
            act2 = act[1]
        change_img = change_color(change_img,act2)
        
        if act==None:
            act3 =  np.random.uniform(args.min_act, args.max_act)
        else:
            act3 = act[2]
        change_img = change_contrast(change_img,act3)
        
        if act==None:
            act4 =  np.random.uniform(args.min_act, args.max_act)
        else:
            act4 = act[3]
        change_img = change_sharpness(change_img,act4)
        
        act=[act1,act2,act3,act4]
    else:
        raise ValueError('Incorrect flag please try from 1 to 5')

    return change_img,act

def preprocess_img(args, image):
    mean = np.mean(image)
    std = np.std(image)
    preprocess = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    image = preprocess(image)
    return torch.FloatTensor(image).to(args.device)

def show_result(img, noisey_img, refined_img):
    cv2.imshow('original image',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.imshow('noisey image', cv2.cvtColor(noisey_img,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.imshow('refined image',cv2.cvtColor(refined_img,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

def get_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou
    
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
    
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
    
def get_F1(truth,pred,iou_threshold):
    TP = 0
    FP = 0
    FN = 0
    dft = pd.DataFrame(truth)
    iou_arr = []
    names = dft[4].unique() #has all the unique class(labels) names
    for i in range(len(names)):
        label_df = dft[dft[4]==names[i]]
        for bbox_orig in np.array(label_df):
            iou_temp = []
            for bbox_pred in pred:
                if len(pred)>0:
                    iou_temp.append(get_iou(bbox_orig[0:4],bbox_pred[0:4].astype(float)))
            if len(pred)>0: # this is because of error when length of pred becomes zero
                overlap = np.max(iou_temp)
                index = np.argmax(iou_temp)
            else:
                overlap=0
            if overlap>iou_threshold:
                if pred[int(index)][-1]==names[i]: # check whether same class or not
                    iou_arr.append(overlap)
                    pred = np.delete(pred,(index),axis=0) #pop out the matched boxes
                    
                    TP+=1
                else:
                    FN+=1
            else:
                FP+=1
    return TP,FP,FN,iou_arr

def plot_image(image_array, DO, classes, colors, image_name):
    DO = DO.detach().cpu().numpy()
    if len(DO)>0:
        for i in range(len(DO)):
            class_id = int(DO[i][-1])
            color = [int(c) for c in colors[class_id]]
            text = '{}:{:.4f}'.format(classes[class_id], DO[i][6])
            cv2.putText(image_array, text, (int(DO[i][1]), int(DO[i][2]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            cv2.rectangle(image_array, (int(DO[i][1]), int(DO[i][2])), (int(DO[i][3]), int(DO[i][4])), color, 2)
        cv2.imshow(image_name, cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

def plot_GT_image(image_array, DO, classes, colors, image_name):
    if len(DO)>0:
        for i in range(len(DO)):
            class_id = classes.index(DO[i][-1])
            color = [int(c) for c in colors[class_id]]
            text = '{}'.format(classes[class_id])
            cv2.putText(image_array, text, (int(DO[i][0]), int(DO[i][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            cv2.rectangle(image_array, (int(DO[i][0]), int(DO[i][1])), (int(DO[i][2]), int(DO[i][3])), color, 2)
        cv2.imshow(image_name, cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

def get_label(data):
    df= pd.read_csv(data)
    class_name = list(df['class'])
    row = [i for i, j in enumerate(class_name) if j == 'hand' or j=='head' or j =='foot']
    df.drop(row, axis = 0, inplace=True)
    return df

def pred_array(detect, classes, voc_classes):
    pred = np.array(detect.cpu())
    pred_arr = []
    for i in range(len(pred)):
        arr = list(pred[i][1:5])
        if classes[int(pred[i][-1])] in voc_classes:
            arr.append(classes[int(pred[i][-1])])
            arr = np.array(arr)
            pred_arr.append(arr)
    pred_arr = np.array(pred_arr)    
    return pred_arr

def get_score(args, resized_gnd_truth_arr, pred_arr, iou_threshold, eps):
    TP,FP,FN,iou = get_F1(resized_gnd_truth_arr, pred_arr, iou_threshold)
    recall = TP/(TP+FN+eps)
    precision = TP/(TP+FP+eps)

    F1 = 2*recall*precision/(precision+recall+eps)
    if len(iou)>0:
        iou_reward = np.mean(iou)
    else:
        iou_reward = 0

    reward = args.alpha*(iou_reward) + (1-args.alpha)*F1
    return reward

def get_image_batch(args, img_path, noise_scale):
    img_batch = torch.zeros(args.batch_size, *(3,128,128)).to(args.device)

    for i in range(args.batch_size):
        image = Image.open(img_path[i])
        act = noise_scale[i]

        if args.action_type=='discrete':
            noisey_img, _ = synthetic_change_dis(args, image, flag = 1, act = act)
        if args.action_type == 'continuous':
            noisey_img, _ = synthetic_change_cont(args, image, flag = 1, act = act)

        noisey_img_array = np.array(noisey_img)
        transform = torchvision.transforms.ToTensor()
        state = transform(noisey_img_array)
        img_batch[i,...] = torch.FloatTensor(state)

    return img_batch
