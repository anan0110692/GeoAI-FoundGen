import os
from os import path
import sys
import dill
import copy
import glob
import random
import time
import math
import pickle

import numpy as np
import scipy
from scipy import io as sio
from collections import OrderedDict
from utili.misc.misc import Adjust_Accelerator_settings,Create_Dataset_Result_Folder,find_pad

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import matplotlib.lines as mlines
from matplotlib import cm
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
from torch.nn.parameter import Parameter
import torch.utils.data as dataf
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize, ToTensor
from torchsummary import summary
import torch.utils.tensorboard as tb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
    Engine,
)
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import (
    ModelCheckpoint,
    EarlyStopping,
    global_step_from_engine,
    Checkpoint,
    DiskSaver,
)

from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
import torchmetrics
import torchgan
import skimage


def eval_metrics(cnn,model_type,Test_loder,DA_Datamodule=None,GPU_NUM=None):
        All_ypre=[]
        All_y=[] 
        for batch in Test_loder:
            x,y=batch
            # out=cnn.cuda(GPU_NUM)(x.cuda(GPU_NUM), False)[0]
            # out=cnn(x,False)[0]
            # x=(x-x.min())/(x.max()-x.min()  )
            
            y_seq=cnn.Fixed_large_model.patchify(y[:,None,None])
            y=cnn.Fixed_large_model.unpatchify(y_seq.clone(),Target=True).squeeze(1).squeeze(1)
        

                

            model_out=cnn(x_t=x.cuda(),mask_ratio_t=0,Seg_output=True)
            # model_out=cnn(x_s=x, mask_ratio_s=0,Seg_output=True)
            
            out=model_out

            out=torch.max(out, 1)[1].squeeze()+1
            
        
            y=torch.squeeze(y)
            out=out[y!=-100]
            y=y[y!=-100]
            All_ypre.append(out)
            All_y.append(y)
        out=torch.cat(All_ypre,0)
        y=torch.cat(All_y,0)

        True_unique_classes_contigous=np.unique(y)
        unique_classes_contigous=True_unique_classes_contigous
        Predicted_unique_classes_contigous=np.unique(out.cpu())
        True_unique_classes_contigous=unique_classes_contigous[unique_classes_contigous!=-100]
        Predicted_unique_classes_contigous=   Predicted_unique_classes_contigous[ Predicted_unique_classes_contigous!=-100]
        Union_unique_classes_contigous= np.union1d(True_unique_classes_contigous,Predicted_unique_classes_contigous)

        


    
        cm = confusion_matrix(y.cpu().detach().numpy().astype(np.int64),
                            out.cpu().detach().numpy(), normalize='true',)
    
        # labels_dip=np.array(['',"building","pervious surface","impervious surface","bare soil","water","coniferous","deciduous","brushwood","vineyard","herbaceous vegetation","agricultural land","plowed land","swimming pool","snow","clear-cut","ligneous","mixed","greenhouse",'others'])
    #------------------------------------------------------------------------------
        labels_dip=DA_Datamodule.classes_labels_array[:-1]
        fig, ax = plt.subplots(figsize=(15, 15))
        # cm_display = ConfusionMatrixDisplay(cm,display_labels=labels_dip[original_classes.astype(np.int64)])
        original_classes=DA_Datamodule.original_unique_classes
        # display_labels_no_map=labels_dip[True_unique_classes_contigous.astype(np.int64)]
        display_labels_with_map=labels_dip[original_classes[True_unique_classes_contigous.astype(np.int64)].astype(np.int64)]
        cm_display=ConfusionMatrixDisplay.from_predictions(y.cpu().detach().numpy().astype(np.int64),out.cpu().detach().numpy(),normalize='true',display_labels=display_labels_with_map,xticks_rotation=45,ax=ax,labels=True_unique_classes_contigous.astype(np.int64))
    #  cm_display = ConfusionMatrixDisplay(cm)
        
        # cm_display.plot(ax=ax)
        plt.show()
    #----------------------------------------------------------------------------------
        #print('Acc='+str(accuracy_score(TS_label[class_indices], ypre[class_indices])))
        print(model_type+" Acc= "+str(balanced_accuracy_score(y.cpu().detach().numpy(), out.cpu().detach().numpy())))
        print(model_type+" mF1= "+str(f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro')))
    
        return balanced_accuracy_score(y.cpu().detach().numpy(), out.cpu().detach().numpy()),cm,jaccard_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro'), f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro'),Union_unique_classes_contigous, f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average=None)

def Plot_confusion(cm_list,model_name,DA_Datamodule=None,list_f1=None,Union_unique_classes_contigous=None):
   
    All_cm=np.mean(np.concatenate(tuple(cm_list),axis=2),2)
    All_f1=np.mean(np.stack(tuple(list_f1)),0)
    labels_dip=DA_Datamodule.classes_labels_array[:]
    original_classes=DA_Datamodule.original_unique_classes
  
    cm_display = ConfusionMatrixDisplay(np.squeeze(All_cm),display_labels=labels_dip[original_classes[Union_unique_classes_contigous.astype(np.int64)].astype(np.int64)])
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(model_name)
    cm_display.plot(ax=ax)
    plt.show()
    return All_cm,All_f1

def inference_map(S_cnn=None,T_cnn=None,Target_test_dataloader=None,DA_Datamodule=None,name=None,GPU_NUM=None):
  
    UH_map = DA_Datamodule. color_map
    Values_all= DA_Datamodule.classes_labels_array
    original_classes=DA_Datamodule.original_unique_classes
    # Target_test_dataloader=DA_Datamodule.Target_dataloader
    GT=torch.tensor(DA_Datamodule.Full_GT_mask)

  
    for i, batch in enumerate(Target_test_dataloader):
          x,y=batch
          fig,ax=plt.subplots(1,x.shape[0],figsize=(20,20),layout="tight",squeeze=False)
          # out=T_cnn.cuda(GPU_NUM)(x.cuda(GPU_NUM), False)[0]
          out=T_cnn.eval()(x_t=x.clone().cuda(),mask_ratio_t=0,Seg_output=True)
          # out=T_cnn.eval()(x_s=x, mask_ratio_s=0,Seg_output=True)
          out=torch.max(out, 1)[1].squeeze()+1
          GT_y=y
          GT_seq=T_cnn.Fixed_large_model.patchify(GT[None,None,None])
          GT_seq_y=T_cnn.Fixed_large_model.patchify(GT_y[None,None])

          GT=T_cnn.Fixed_large_model.unpatchify(GT_seq.clone(),Target=True).squeeze(1).squeeze(1)
          GT_y=T_cnn.Fixed_large_model.unpatchify(GT_seq_y.clone(),Target=True).squeeze(1).squeeze(1)
          GT[GT==-100]=0
          GT_y[GT_y==-100]=0
          # mapped_out_to_original_classes=GT

          mapped_out_to_original_classes=original_classes[out.detach().cpu().numpy().astype(np.int64)]
          # mapped_out_to_original_classes=original_classes[GT.squeeze().detach().cpu().numpy().astype(np.int64)]
          mapped_out_to_original_classes[torch.squeeze( GT)== 0]=0
          # mapped_out_to_original_classes=original_classes[GT_y.squeeze().detach().cpu().numpy().astype(np.int64)]
          # mapped_out_to_original_classes[torch.squeeze( GT_y)== 0]=0
          
          
         
          im=ax[0,i].imshow(mapped_out_to_original_classes.squeeze().astype(np.int64),cmap=UH_map,norm=matplotlib.colors.NoNorm())
         
          ax[0,i].axis('off')
         
          selected_original_classes=original_classes[np.unique(GT.detach().cpu().numpy().astype(np.int64))]#---
          selected_original_classes[selected_original_classes==-100]=Values_all.size-1
          Selected_Values=Values_all[selected_original_classes.astype(np.int64)]
          Values_true= Selected_Values
          fig.savefig('Inference_map_'+name+'.png',dpi=300,bbox_inches='tight',format='png') 
        
      


    plt.show()