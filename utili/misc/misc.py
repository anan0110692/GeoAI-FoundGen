#! pip install -r /big-data/GeoAI/users/anan/experiments/cleanDA/requirements.txt
#from unet4 import UNet
import os
# from os import path
# import sys
import dill
import copy
import glob
# import random
# import time
# import math
# import pickle

import numpy as np
# import scipy
# from scipy import io as sio
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.markers as mmarkers
import matplotlib.lines as mlines
# from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color

from sklearn.utils import shuffle, check_random_state
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import torch.utils.data as dataf
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize, ToTensor
# from torchsummary import summary
import torch.utils.tensorboard as tb



from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning  as L

# from PIL import Image
# from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
# import torchmetrics
import torchgan
# import skimage

# sys.path.append("../")
# torch.set_float32_matmul_precision('medium')
# torch.backends.cudnn.allow_tf32=True




def Adjust_Accelerator_settings(GPU_NUM=None):
   
        global  devices_S
        global  accelerator_type_S
        global  num_workers_S
        if GPU_NUM is not  None:
            devices_S =[GPU_NUM]
            num_workers_S=0
            accelerator_type_S='gpu'
            devices_DA=[0]
            num_workers_DA=0
            accelerator_type_DA='gpu'
        else:
            accelerator_type_S='cpu'
            accelerator_type_DA='cpu'
            num_workers_S=0
            num_workers_DA=0
            devices_S =[0]
            devices_DA=[0]
      
        
       
        return devices_S, num_workers_S, accelerator_type_S,devices_DA, num_workers_DA, accelerator_type_DA

# Section 4: Results Folder Creation
# This function creates a folder structure for storing results.
# It creates a main directory for the dataset, and two subdirectories for Source and DA results.

def Create_Dataset_Result_Folder(Dataset,Exp_name):
    Dataset_name=Dataset.Dataset_name
    Result_Folder='Results'
    Dataset_path=os.path.join(Result_Folder,Dataset_name)
    Exp_num_subfolder=os.path.join(Dataset_path,"Exp_"+Exp_name)
    Source_subfolder=os.path.join(Dataset_path,"Source","Exp_"+Exp_name)
    DA_subfolder=os.path.join(Dataset_path,"DA","Exp_"+Exp_name)
    lightning_logs_subfolder=os.path.join(Dataset_path,"lightning_logs")
    
    if not os.path.exists(Dataset_path):
        os.makedirs(Dataset_path)
        os.makedirs(Exp_num_subfolder)
        os.makedirs(Source_subfolder)
        os.makedirs(DA_subfolder)
        os.makedirs(lightning_logs_subfolder)
        
    else:
        if not os.path.exists(Source_subfolder):
            os.makedirs(Source_subfolder)
        if not os.path.exists(DA_subfolder):
            os.makedirs(DA_subfolder)
        if not os.path.exists(Exp_num_subfolder):
            os.makedirs(Exp_num_subfolder)
        if not os.path.exists(lightning_logs_subfolder):
            os.makedirs(lightning_logs_subfolder)
       
    return Exp_num_subfolder,lightning_logs_subfolder,Dataset_path,Source_subfolder,DA_subfolder

def Xavi_init_weights(model):
    if isinstance(model, nn.Module):
        for name, param in model.named_parameters():
            if 'weight' in name and not 'norm' in name and not 'Batch1' in name and not 'Batch2' in name and  not 'Batch3' in name:
                if 'enc' in name :
                    torch.nn.init.xavier_normal_(param.data, gain=torch.nn.init.calculate_gain('relu'))
                else:
                    torch.nn.init.xavier_normal_(param.data)
def set_parameter_requires_grad(model):
  
        for name,mod in model.named_children():
            
            if name[0]=='_' or name=="final_layer":
                   for param in mod.parameters():
                        param.requires_grad = False

def get_param(model):
    model.requires_grad = True
    set_parameter_requires_grad(model)
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


@torch.no_grad()
def find_pad(model,x):
       
        name_list=[]
      
        for name,_ in model.named_children():
            if "upconv" in name:
                name_list.append(name)
        for i in range(len(name_list)):
            temp=getattr(model,name_list[i])
            if isinstance(temp,torchgan.layers.spectralnorm.SpectralNorm2d):
                
                modell=temp.module
                if not hasattr(modell,'weight'):
                    temp_x=torch.rand((3,modell.in_channels,100,100),device=x.device)
                    temp_out=temp(temp_x)
                modell=temp.module
                setattr(modell,'output_padding',(0,0))
            else:
                setattr(temp,'output_padding',(0,0))
       
        encoders_output_shapes=[]
        decoders_output_shapes=[]
        out_sizes_list=[]
        out=x
        for name,Sub_model in model.named_children():
           
            if ~("final_layer" in name) and len(decoders_output_shapes)==0 :
                # out=Sub_model.cuda(GPU_NUM)(out)
                # out=Sub_model(out)
                ###################################################################
                if len(out_sizes_list)==0:
                        input_shape=x.shape
                else:
                        input_shape=out_sizes_list[-1]
              
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)


               ######################################################################
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])
            else:
                if "upconv" in name:
                    input_shape=encoders_output_shapes[-len(decoders_output_shapes)]
                   
                    
                     
                else:
                    input_shape= out_sizes_list[-1]
                    # out=Sub_model.cuda(GPU_NUM)(out)
                    if "final" in name:
                       continue
                    
                if isinstance(Sub_model,torch.nn.modules.container.Sequential):
                   out_shape=input_shape
                   for i in Sub_model:
                       if isinstance(i,torch.nn.modules.conv.Conv2d):
                           Last_conv_layer=i
              
                   out_shape=tuple(input_shape[0], Last_conv_layer.out_channels,*input_shape[2:])
                   out_sizes_list.append(out_shape)
               
               #-------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,torch.nn.modules.pooling.MaxPool2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))
                    # D_temp= torch.tensor((input_shape[-3]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    # D=torch.floor(D_temp)

                    H_temp= torch.tensor((input_shape[-2]+2*Padding_size[0]-  Dilation_size[0]*(Kernal_size[0]-1)-1)/ Stride_size[0]+1)
                    H=torch.floor(H_temp)
                    
                    W_temp= torch.tensor((input_shape[-1]+2*Padding_size[1]-  Dilation_size[1]*(Kernal_size[1]-1)-1)/ Stride_size[1]+1)
                    W=torch.floor(W_temp)

                    out_shape=tuple(input_shape[0],input_shape[1] ,H,W)
                    out_sizes_list.append(out_shape)

                    #--------------------------------------------------------------------------------------------------------------
                elif isinstance(Sub_model,nn.ConvTranspose2d):
                    Kernal_size=Sub_model.kernel_size
                    if isinstance(Kernal_size,int):
                        Kernal_size=(torch.tensor(Kernal_size),torch.tensor(Kernal_size))
                   
                    Padding_size=Sub_model.padding
                    if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size))

                    Stride_size=Sub_model.stride
                    if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size))
                    
                   
                    Dilation_size=Sub_model.dilation
                    if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size))

                    
                    # D=(input_shape[-3]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    H=(input_shape[-2]-1)*Stride_size[0]-2*Padding_size[0]+Dilation_size[0]*( Kernal_size[0]-1)+1
                    W=(input_shape[-1]-1)*Stride_size[1]-2*Padding_size[1]+Dilation_size[1]*( Kernal_size[1]-1)+1
                    out_shape=(input_shape[0], Sub_model.out_channels,H,W)
                    out_sizes_list.append(out_shape)
                    #---------------------------------------------------------------------------------------------
                elif  isinstance(Sub_model,nn.Conv2d):
                    out_shape=(input_shape[0], Sub_model.out_channels,*input_shape[2:])
                    out_sizes_list.append(out_shape)
                #----------------------------------------------------------------------------
                else:
                    out_shape=input_shape
                    out_sizes_list.append(out_shape)
              
              
               
              
              
                if "encoder" in name:
                    encoders_output_shapes.append(out_sizes_list[-1])
                elif "upconv" in name:
                    decoders_output_shapes.append(out_sizes_list[-1])

    
        decoders_output_shapes.reverse()
        output_padding=[]

        for ii in range(len(decoders_output_shapes)):
            temp_enc_shape= encoders_output_shapes[ii]
            temp_dec_shape= decoders_output_shapes[ii]
            output_padding.append((int( temp_enc_shape[-2]-temp_dec_shape[-2]),int( temp_enc_shape[-1]-temp_dec_shape[-1])))
        output_padding.reverse()
        return output_padding