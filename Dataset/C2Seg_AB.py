# %%

##################### Import Libraries#############################
#! pip install -r /big-data/GeoAI/users/anan/experiments/cleanDA/requirements.txt
#from unet4 import UNet

global Update_u
import glob
import dill
# sys.path.append("../")
import torchvision
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from collections import OrderedDict
import os
from os import path
import numpy as np
import random
import matplotlib
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from scipy import io as sio
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from torchsummary import summary
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import copy
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
import math
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy import io as sio
import scipy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as dataf
import torch
import matplotlib
import random
import numpy as np
from os import path
import os
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../")
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import matplotlib.markers as mmarkers
import matplotlib.lines as mlines
##################### Import Libraries#############################
#! pip install -r /big-data/GeoAI/users/anan/experiments/cleanDA/requirements.txt
#from unet4 import UNet
import dill
# sys.path.append("../")
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
from collections import OrderedDict
import os
from os import path
import numpy as np
import random
import matplotlib
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from scipy import io as sio
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from torchsummary import summary
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import copy
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
import math
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
from sklearn.decomposition import PCA
from scipy import io as sio
import scipy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as dataf
import torch
import matplotlib
import random
import numpy as np
from os import path
import os
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, cohen_kappa_score,balanced_accuracy_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../")
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator,Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine, Checkpoint,DiskSaver
import pickle
import lightning  as L

from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import jaccard_score
from matplotlib.colors import hex2color
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.fid import _compute_fid
from torchvision.transforms import Normalize
import torchmetrics
import torchgan
# import skimage
import torch.utils.tensorboard as tb
from torchvision.transforms import ToTensor
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
import torch.utils.tensorboard as tb
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from einops import rearrange
from prithvi.Prithvi import MaskedAutoencoderViT
import lightning  as L
from functools import partial

import torch
import torch.nn as nn

# from timm.models.vision_transformer import Block
# from timm.models.layers import to_2tuple

import numpy as np

from scipy import signal
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, cohen_kappa_score,balanced_accuracy_score




# %%




def mapping_mask(mask):
    mapped_mask=np.ones_like(mask)*-100   
    unique_classes=np.unique(mask)
    mapped_classes=np.zeros((unique_classes.max().astype(np.int64)+1,))
    mapped_classes[unique_classes[unique_classes!=-100].astype(np.int64)]=np.arange(1,unique_classes[unique_classes!=-100].size+1)
    mapped_mask[mask!=-100]=mapped_classes[mask[mask!=-100].astype(np.int64)]
    
    return mapped_mask

# %%
Dataset_name='C2Seg_AB'
def select_well_represented_common_classes(split=True):
    Source_Path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datafiles','C2Seg_AB','berlin_multimodal.mat')
    Source_data = sio.loadmat(Source_Path)
    Source_cube=Source_data['HSI']
    Source_cube = ( Source_cube.astype(np.float32))   
    

   

    ##################################################################3
    Source_label =  Source_data['label']
    Source_label = Source_label.astype(np.float32)
    Source_DATA_dic={'raw':Source_cube,'label':Source_label}


    Target_Path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datafiles','C2Seg_AB','augsburg_multimodal.mat')
    Target_data = sio.loadmat(Target_Path)
    Target_cube=Target_data['HSI']
    Target_cube = Target_cube.astype(np.float32)
   


    ##################################################################3
   
    
   
    Target_label =  Target_data['label']
    Target_label = Target_label.astype(np.float32)
    Target_DATA_dic={'raw':Target_cube,'label':Target_label}

   
   

    Source_unique_classes,Source_histo= np.unique(Source_DATA_dic['label'],return_counts=True)
    Source_mask= Source_DATA_dic['label'].astype(np.int64)
    Target_mask= Target_DATA_dic['label'].astype(np.int64)
    Target_unique_classes,Target_histo=  np.unique(Target_DATA_dic['label'],return_counts=True)
    common_classes,S_common_idx,T_common_idx= np.intersect1d(Source_unique_classes,Target_unique_classes,assume_unique=True,return_indices=True)
    common_from_S_histo= Source_histo[S_common_idx]
    common_from_S=     Source_unique_classes[S_common_idx]
    Underrepresend_classes=  common_from_S[common_from_S_histo<np.quantile(common_from_S_histo,.3) ]
    if  not split:
        Source_mask[np.isin(Source_mask.copy(),Underrepresend_classes)]=-100
        Target_mask[np.isin(Target_mask.copy(),Underrepresend_classes)]=-100
        return Source_DATA_dic['Mosaic_image'],Source_mask, Target_DATA_dic['Mosaic_image'],Target_mask
    else:
        common_classes,S_common_idx,T_common_idx= np.intersect1d(Source_unique_classes,Target_unique_classes,assume_unique=True,return_indices=True)

       
        
        return Source_DATA_dic['raw'],Source_DATA_dic['label'], Target_DATA_dic['raw'], Target_DATA_dic['label'] , common_classes.size, None

# %%
select_well_represented_common_classes(split=True)[-1]

# %%
def clnum():
    
    return select_well_represented_common_classes(split=True)[4]

# %%
def Source_data_generator(batch_size=1,num_workers=16):
    Source_dataset=select_well_represented_common_classes(split=True)
    # HU_cube=preprocess_image(Source_dataset[0][:,0:224].copy().astype(np.float32)/(pow(2,14)-1)).astype(np.float32)
    HU_cube=Source_dataset[0][0:1000,:500,2:].copy().astype(np.float32)
    HU_cube_T=Source_dataset[2][:,:].copy().astype(np.float32)/(pow(2,14)-1)
    HU_cube = HU_cube[None]
    HU_cube_T = HU_cube_T[None]
    HU_cube=(HU_cube-HU_cube.min())/(HU_cube.max()-HU_cube.min())
    HU_cube_T=(HU_cube_T-HU_cube_T.min())/(HU_cube_T.max()-HU_cube_T.min())
    # HU_cube_resampled=Source_dataset[-1][None,:,None,:100,:100].astype(np.float32)/(pow(2,14)-1)
    # HU_cube_resampled=(HU_cube_resampled-HU_cube_resampled.min())/(HU_cube_resampled.max()-HU_cube_resampled.min())
    # HU_cube_resampled=torch.from_numpy(HU_cube_resampled)
    
    HU_cube=torch.from_numpy(HU_cube[:,:,:,:]).permute((0,3,1,2))
    HU_cube_T=torch.from_numpy(HU_cube_T[:,:,:,:]).permute((0,3,1,2))
    # HU_cube=torch.from_numpy(HU_cube)
    ch_num=HU_cube.shape[1]
    ch_num_T=HU_cube_T.shape[1]
    TR_label=Source_dataset[1][:1000,:500]
    
    TR_label = TR_label.astype(np.float32)
    TR_label[TR_label==0]= -100
    original_unique_classes=np.unique(TR_label)
    TR_label= mapping_mask(TR_label)
    TR_label=np.squeeze(TR_label)
    cl_num = int(np.max(TR_label))
    source_dataset_seed=41
    # seed_everything(source_dataset_seed,workers=True)
    # random_state = check_random_state(source_dataset_seed)
    ######################################################################################33
    All_Train_labels_idx = np.array([])
    All_Val_labels_idx = np.array([])
    classes = np.unique(  TR_label.flatten())
    classes = classes[classes != -100]
    #classes = classes[classes == 1]
    for i in classes:
        class_indices = np.argwhere(TR_label == i)
        class_train_idx, class_val_idx, _, _ = train_test_split(class_indices, np.zeros(
            (class_indices.shape[0],)), train_size=.8, random_state=42)
        if All_Train_labels_idx.size == 0:
            All_Train_labels_idx = class_train_idx
        else:
            All_Train_labels_idx = np.concatenate(
                (All_Train_labels_idx, class_train_idx))

        if All_Val_labels_idx.size == 0:
            All_Val_labels_idx = class_val_idx
        else:
            All_Val_labels_idx = np.concatenate(
                (All_Val_labels_idx, class_val_idx))


   


    TR_labell= np.ones_like(TR_label)*-100
    
    VA_label = np.ones_like(TR_label)*-100
    VA_label[All_Val_labels_idx[:, 0], All_Val_labels_idx[:, 1]] = TR_label[All_Val_labels_idx[:, 0], All_Val_labels_idx[:, 1]]
    TR_labell[All_Train_labels_idx[:, 0], All_Train_labels_idx[:, 1]] = TR_label[All_Train_labels_idx[:, 0], All_Train_labels_idx[:, 1]]
    TR_labell=mapping_mask(TR_labell)
    TR_labell = torch.from_numpy(TR_labell)
    TR_labell = TR_labell[None, :, :]
    VA_label=mapping_mask(VA_label)
    VA_label = torch.from_numpy(VA_label)
    VA_label = VA_label[None, :, :]
    dataset_T = dataf.TensorDataset(HU_cube[:,:], TR_labell)
    train_loader = dataf.DataLoader(dataset_T, batch_size=batch_size,num_workers=num_workers)
    dataset_V = dataf.TensorDataset(HU_cube, VA_label)
    Validation_loader = dataf.DataLoader(dataset_V, batch_size=batch_size,num_workers=num_workers)
    return train_loader, Validation_loader,original_unique_classes,ch_num, (Source_dataset[0][:100,:100].copy().astype(np.float32)/(pow(2,14)-1)).transpose(2,0,1),ch_num_T,TR_label

# %%
def Target_data_genrator(batch_size=1, num_workers=16,Num_of_Samples=None):
   
    
    Source_dataset=select_well_represented_common_classes(split=True)

    HU_cube = Source_dataset[2][:500,:1000,:].copy().astype(np.float32)/(pow(2,14)-1)
    HU_cube = HU_cube[None]
    
    HU_cube=(HU_cube-HU_cube.min())/(HU_cube.max()-HU_cube.min())
    ch_num=HU_cube.shape[-1]
    HU_cube=torch.from_numpy(HU_cube[:,:,:,:]).permute((0,3,1,2))
   
   
   
   
    TR_label=Source_dataset[3][:500,:1000]
    
    TR_label = TR_label.astype(np.float32)
    TR_label[TR_label==0]= -100
    original_unique_classes=np.unique(TR_label)
    TR_label= mapping_mask(TR_label)
    TR_label=np.squeeze(TR_label)
    cl_num = int(np.max(TR_label))

    target_dataset_seed=41
    seed_everything(target_dataset_seed,workers=True)
    random_state = check_random_state(target_dataset_seed)
    #######################################################################################
    All_Train_labels_unlabeled_idx = np.array([])
    All_Train_labels_labeled_idx=np.array([])
    All_Val_labels_labeled_idx=np.array([])
    All_test_labels_idx = np.array([])
    # with open('/project/prasad/NASA_FOUNDATATIONAL_JOINT_FOR_CARYA/hls-foundation-os/Dataset/Fixed_Val_dataloader/Fixed_Germany_wishpers_MOSAIC_All_Normalization_test_labels.pkl', 'rb') as file:
    #        All_Val_labels_labeled_idx= dill.load(file)
    # # # All_test_labels_idx_dict={tuple(row): i for i, row in enumerate(All_test_labels_idx)}
    # with open('/project/prasad/NASA_FOUNDATATIONAL_JOINT_FOR_CARYA/hls-foundation-os/Dataset/Fixed_Test_dataloader/Fixed_Germany_wishpers_MOSAIC_All_Normalization_test_labels.pkl', 'rb') as file:
    #        All_test_labels_idx= dill.load(file)
  
    # All_Val_test_labels_idx=np.concatenate((All_Val_labels_labeled_idx,All_test_labels_idx))
    # # All_Val_labels_idx=((All_Val_labels_labeled_idx))
    # All_Val_test_labels_idx_dict={tuple(row): i for i, row in enumerate(All_Val_test_labels_idx)}
    classes = np.unique(  TR_label.flatten())
    classes = classes[classes != -100]
   # classes = classes[(classes == 13)]
    for i in classes:
        class_indices = np.argwhere( TR_label  == i)
        class_indice_copy= class_indices.copy()
        # selected_class_indices_idx= [ All_test_labels_idx_dict.get(tuple(row))==None for i,row in enumerate(class_indices)]
        # selected_Val_class_indices_idx= [  All_Val_test_labels_idx_dict.get(tuple(row))==None for i,row in enumerate(class_indices)]
        # class_indices=class_indices[selected_Val_class_indices_idx]
        if Num_of_Samples is not None:
           
           



        # class_train_unlabeled_idx=class_indices
        # class_train_idx=class_indices
       
            if i==8 or i==12 or i==13   :
                class_train_idx, class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=Num_of_Samples, random_state=42)
            
            
            elif i !=2 and i!=3:
                
                class_train_idx,  class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=Num_of_Samples, random_state=42)
            else:
                class_train_idx,  class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=Num_of_Samples, random_state=42)
        # class_train_idx=  class_indices 
        else:
            
            if i==8 or i==12 or i==13   :
                class_train_idx, class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=.7, random_state=42)
            
            
            elif i !=2 and i!=3:
                
                class_train_idx,  class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=2000, random_state=42)
            else:
                class_train_idx,  class_test_idx, _, _ = train_test_split(class_indices, np.zeros((class_indices.shape[0],)), train_size=2000, random_state=42)

        if i==8 or i==12 or i==13:
            class_test_idx,class_val_labeled_idx,_,_= train_test_split( class_test_idx, np.zeros(
            ( class_test_idx.shape[0],)), train_size=2/3, random_state=42)
        else:
            class_test_idx,class_val_labeled_idx,_,_= train_test_split( class_test_idx, np.zeros(
            ( class_test_idx.shape[0],)), train_size=.5, random_state=42)
        
       
        # All_test_labels_idx_dict={tuple(row): i for i, row in enumerate(class_No_train_idx)}
        # selected_class_indices_idx= [ All_test_labels_idx_dict.get(tuple(row))==None for i,row in enumerate(class_indices)]
        # class_indices=class_indices[ selected_class_indices_idx]
        # class_train_idx, _, _, _ = train_test_split(class_indice_copy, np.zeros((class_indice_copy.shape[0],)), train_size=Num_of_Samples, random_state=42)
        # class_train_unlabeled_idx,class_val_labeled_idx,_,_= train_test_split(class_train_unlabeled_idx, np.zeros(
        #     (class_train_unlabeled_idx.shape[0],)), train_size=.8, random_state=42)
        
        if All_Train_labels_unlabeled_idx.size == 0:
            All_Train_labels_unlabeled_idx = class_train_idx
        else:
            All_Train_labels_unlabeled_idx = np.concatenate(
                (All_Train_labels_unlabeled_idx, class_train_idx))
       
        # if  All_Train_labels_labeled_idx.size == 0:
        #     All_Train_labels_labeled_idx= class_train_labeled_idx
        # else:
        #   All_Train_labels_labeled_idx = np.concatenate(
        #         (All_Train_labels_labeled_idx, class_train_labeled_idx))
       #######################################################################
        if  All_Val_labels_labeled_idx.size == 0:
           All_Val_labels_labeled_idx= class_val_labeled_idx
        else:
           All_Val_labels_labeled_idx = np.concatenate(
                ( All_Val_labels_labeled_idx, class_val_labeled_idx))


       
        if All_test_labels_idx.size == 0:
            All_test_labels_idx = class_test_idx
        else:
            All_test_labels_idx = np.concatenate(
                (All_test_labels_idx, class_test_idx))
     
    TE_label = np.ones_like(TR_label)*-100
    TR_labell=  np.ones_like(TR_label)*-100
    # TR_label_labeled=  np.ones_like(TR_label)*-100
    VA_label_labeled=  np.ones_like(TR_label)*-100
    TE_label [All_test_labels_idx[:, 0], All_test_labels_idx[:, 1]] =  TR_label[All_test_labels_idx[:, 0], All_test_labels_idx[:, 1]]
    TR_labell[All_Train_labels_unlabeled_idx[:, 0], All_Train_labels_unlabeled_idx[:, 1]] =  TR_label[All_Train_labels_unlabeled_idx[:, 0], All_Train_labels_unlabeled_idx[:, 1]]
    # TR_label_labeled[All_Train_labels_labeled_idx[:, 0], All_Train_labels_labeled_idx[:, 1]] =  TR_label[All_Train_labels_labeled_idx[:, 0], All_Train_labels_labeled_idx[:, 1]]
    VA_label_labeled[All_Val_labels_labeled_idx[:, 0], All_Val_labels_labeled_idx[:, 1]] =  TR_label[All_Val_labels_labeled_idx[:, 0], All_Val_labels_labeled_idx[:, 1]]

   

   
  
   
    TR_labell  = torch.from_numpy(TR_labell )
    TR_labell  = TR_labell [None, :, :]
    TE_label = torch.from_numpy(TE_label)
    TE_label = TE_label[None, :, :]
    # TR_label_labeled  = torch.from_numpy(TR_label_labeled)
    # TR_label_labeled  = TR_label_labeled [None, :, :]
    VA_label_labeled  = torch.from_numpy(VA_label_labeled)
    VA_label_labeled  = VA_label_labeled [None, :, :]
   
    dataset_T = dataf.TensorDataset(HU_cube,TR_labell )
    train_loader = dataf.DataLoader(dataset_T, batch_size=batch_size,num_workers=num_workers)
    dataset_TE = dataf.TensorDataset(HU_cube, TE_label)
    TEST_loader = dataf.DataLoader(dataset_TE,  batch_size=batch_size,num_workers=num_workers)
    # dataset_train_labeled = dataf.TensorDataset(HU_cube,TR_label_labeled )
    # train_labeled_loader= dataf.DataLoader(  dataset_train_labeled,  batch_size=batch_size,num_workers=num_workers)
    dataset_val_labeled = dataf.TensorDataset(HU_cube,VA_label_labeled )
    val_labeled_loader= dataf.DataLoader(  dataset_val_labeled,  batch_size=batch_size,num_workers=num_workers)
    train_unlabeled_loader=train_loader




        
    

   


 
 
 
 
 
 
 
    return train_unlabeled_loader, TEST_loader, val_labeled_loader, TR_label 
    # return train_unlabeled_loader, TEST_loader, val_labeled_loader,All_test_labels_idx

# %%
def Color_map_generator():
   
   
    lut_colors = {
    0 : '000000',
    1  : '#01fdfd',
    2  : '#fcfbfb',
    3  : '#fc0101',
    4  : '#dda0dc',
    5  : '#8f04cc',
    6  : '#ff83fe',
    7  : '#ffdd83',
    8  : '#ca8540',
    9  : '#bdb76b',
    10 : '#01fc01',
    11 : '#9acc33',
    12 : '#8a4413',
    13 : '#826ffd'
}
 
   

    
   
    Color_map_array=[list(int(i * 255) for i in hex2color(v)) for k, v in lut_colors.items()]
    Color_map_array=np.array(Color_map_array)
   
    UH_map = ListedColormap(Color_map_array.astype(np.float32)/256.0)
    
    return UH_map

# %%
class Source_Datamodule(L.LightningDataModule):
    
    def __init__(self, batch_size=3, num_workers=16  ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        Source_data_generator_Output_tuple=Source_data_generator(self.batch_size, self.num_workers)
        # Target_data_generator_output_tuple=Target_data_genrator(self.batch_size, self.num_workers)
        self.train_loader=Source_data_generator_Output_tuple[0]
        # self.Target_train_loader=Target_data_generator_output_tuple[2]
        # self.Target_val_loader=Target_data_generator_output_tuple[3]
        self.val_loader=Source_data_generator_Output_tuple[1]
        self.original_unique_classes=Source_data_generator_Output_tuple[2]
        self.original_imge=Source_data_generator_Output_tuple[-1]
        

        self.classes_labels_array= np.array( [' ',"Surface water", "Street", "Urban Fabric", "Industrial, commercial and transport", "Mine, dump, and construction sites", "Artificial, vegetated areas", "Arable Land", "Permanent Crops", "Pastures", "Forests", "Shrub", "Open spaces with no vegetation", "Inland wetlands" ,'ignored']
)
        self.color_map=Color_map_generator()
        self.ch_num=Source_data_generator_Output_tuple[3]
        self.cl_num_T=Source_data_generator_Output_tuple[-2]
        self.cl_num=self.original_unique_classes[self.original_unique_classes!=-100].size
        self.Full_GT_mask=  Source_data_generator_Output_tuple[-1]
    def setup(self, stage=None):
       pass
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader

# %%
class Target_Datamodule(L.LightningDataModule):
    
    def __init__(self, batch_size=1, num_workers=16 ,Num_of_Samples=None ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        Target_data_generator_Output_tuple=Target_data_genrator(self.batch_size, self.num_workers,Num_of_Samples=Num_of_Samples)
        self.train_loader=Target_data_generator_Output_tuple[0]
        self.test_loader=Target_data_generator_Output_tuple[1]
        self.val_loader= Target_data_generator_Output_tuple[-2]
        self.Full_GT_mask=  Target_data_generator_Output_tuple[-1]
    def setup(self, stage=None):
       pass
    def train_dataloader(self):
        return self.train_loader
    def test_dataloader(self):
        return self.test_loader
    def val_dataloader(self):
        return self.val_loader

# %%
class DA_Datamodule(L.LightningDataModule):
        
        def __init__(self, batch_size=1, num_workers=16, Source_Datamodule_arg=None, Num_of_Samples=None  ):
            super().__init__()
            self.batch_size = batch_size
            self.num_workers = num_workers

            if Source_Datamodule_arg is None:
                _Source_Datamodule=Source_Datamodule(self.batch_size, self.num_workers)
            else:
                _Source_Datamodule=Source_Datamodule_arg
            _Target_Datamodule=Target_Datamodule(self.batch_size, self.num_workers,Num_of_Samples=Num_of_Samples)
            self.ch_num=_Source_Datamodule.ch_num
            self.Source_dataloader=_Source_Datamodule.train_dataloader()
            self.Target_dataloader=[]
            
            self.Target_dataloader=_Target_Datamodule.train_dataloader()
            self.Target_test_dataloader=_Target_Datamodule.test_dataloader()
            self.Target_val_dataloader= _Target_Datamodule.val_dataloader()
            # self.Target_val_dataloader=_Target_Datamodule.test_dataloader()
            self.classes_labels_array=_Source_Datamodule.classes_labels_array
            self.color_map=Color_map_generator()
            self.Total_class_num= _Source_Datamodule.classes_labels_array.size
            self.cl_num=_Source_Datamodule.cl_num
            self.ch_num=_Source_Datamodule.ch_num
            self.Full_GT_mask=_Target_Datamodule.Full_GT_mask
            self.original_unique_classes=_Source_Datamodule.original_unique_classes
        def setup(self, stage=None):
            pass
        def train_dataloader(self):
            return [self.Target_dataloader, self.Source_dataloader]
        def val_dataloader(self) :
            return self.Target_val_dataloader
        def get_batch(self,batch_Datset='Target'):
            DA_loader=self.train_dataloader()
            batch_list=next(iter(DA_loader))
            if batch_Datset=='Target':
                return  next(iter(DA_loader[0]))
            else:
                return  next(iter(DA_loader[1]))






#%% Model Definition 


def load_pre_trained_model(S_input_shape,T_input_shape,random_initi=False):
    weights_path = os.path.join(os.getcwd(),'prithvi','Prithvi_100M.pt')
    checkpoint = torch.load(weights_path, map_location="cpu")

# read model config
    model_cfg_path = os.path.join(os.getcwd(),'prithvi','Prithvi_100M_config.yaml') 
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]

# let us use only 1 frame for now (the model was trained on 3 frames)
    model_args["num_frames"] = 1
    # model_args["patch_size"]=2
    model_args["in_chans"] = 6
    model_args["img_size"] =  S_input_shape[-2:]
    model_args["T_img_size"] = T_input_shape[-2:]
    





# instantiate model
    model = MaskedAutoencoderViT(**model_args)
    model.eval()

    # load weights into model
    # strict=false since we are loading with only 1 frame, but the warning is expected
    del checkpoint['pos_embed']
    del checkpoint['decoder_pos_embed']
    
    if not random_initi:
        _ = model.load_state_dict(checkpoint, strict=False)
    return model,model_args



class Seg_Conv_Blender(nn.Module):
 def __init__(self, dummy_feature_shape,S_input_shape,T_input_shape,num_cl,num_feature=512):
  super(Seg_Conv_Blender, self).__init__()
  temp=load_pre_trained_model(S_input_shape,T_input_shape,random_initi=True)
  self.model_args=temp[-1]
  self.Fixed_large_model=temp[0]
  input_shape=dummy_feature_shape
  Belender_Seq_length= self.Fixed_large_model.patch_embed.num_patches+self.Fixed_large_model.patch_embed_T.num_patches +2
  Blender_stride= Belender_Seq_length//self.Fixed_large_model.patch_embed_T.num_patches
  Blender_kernal=Belender_Seq_length-(self.Fixed_large_model.patch_embed_T.num_patches-1)*Blender_stride
  
  self.Blender=nn.Conv1d(input_shape[1],input_shape[1],Blender_kernal,Blender_stride)
#   self.conv_0= nn.Conv2d(input_shape[1], num_feature,3,padding='same')
#   self.Batch_Norm_0=nn.BatchNorm2d(num_feature)
  self.deconv1= nn.ConvTranspose2d(num_feature, num_feature, 2, stride=2)
  self.conv1=nn.Conv2d(num_feature,num_feature//2,2,padding='same')
  self.Batch_Norm_1=nn.BatchNorm2d(num_feature//2)
  self.deconv2= nn.ConvTranspose2d(num_feature//2, num_feature//2, 2, stride=2)
  self.conv2=nn.Conv2d(num_feature//2,num_feature//4,2,padding='same')
  self.Batch_Norm_2=nn.BatchNorm2d(num_feature//4)
  self.deconv3= nn.ConvTranspose2d(num_feature//4, num_feature//4, 2, stride=2)
  self.conv3=nn.Conv2d(num_feature//4,num_feature//8,1,padding='same')
  self.Batch_Norm_3=nn.BatchNorm2d(num_feature//8)
  self.deconv4= nn.ConvTranspose2d(num_feature//8, num_feature//8, 2, stride=2)
#   self.conv4_Seg=nn.Conv2d((num_feature//8+6),num_feature//16,1,padding='same')
  self.conv4=nn.Conv2d((num_feature//8),num_feature//16,3,padding='same')
  self.Batch_Norm_4=nn.BatchNorm2d(num_feature//16)
  self.final_classifyer=nn.Conv2d(num_feature//16,num_cl,1)
  self.final_MAE_head=nn.Conv2d(num_feature//16, T_input_shape[1],1)
  self.MAE_Dec=None
  
 def forward(self, x_seq, id_restore_s=None, id_restore_t=None ,y=None,MAE=False):
        self.MAE_Dec=MAE
        if not self.MAE_Dec:   
          if id_restore_t is not None:
               x_feature_map=torch.reshape(x_seq[:,1:].flatten(),(x_seq.shape[0],self.model_args["T_img_size"][0]//self.model_args["patch_size"],self.model_args["T_img_size"][1]//self.model_args["patch_size"],-1)).permute(0,-1,1,2)
             
          else:
               x_feature_map=torch.reshape(x_seq[:,1:].flatten(),(x_seq.shape[0],self.model_args["img_size"][0]//self.model_args["patch_size"],self.model_args["img_size"][1]//self.model_args["patch_size"],-1)).permute(0,-1,1,2)
               
        else:
              random_indices = torch.randperm(x_seq.shape[1], generator=torch.Generator().manual_seed(50))
              # random_indices = torch.randperm(x_seq.shape[1])
              x_seq_shfulled=x_seq[:,random_indices]
              belender_out=self.Blender(x_seq_shfulled.permute((0,-1,1))).permute((0,-1,1))
              x_feature_map=torch.reshape(belender_out.flatten(),(x_seq.shape[0],self.model_args["T_img_size"][0]//self.model_args["patch_size"],self.model_args["T_img_size"][1]//self.model_args["patch_size"],-1)).permute(0,-1,1,2)
        # out3=self.forward_layer_last(out3)
       #  out_img=self.conv_0(x_feature_map)
       #  out_img=self.Batch_Norm_0(out_img)
        out_img=self.deconv1(x_feature_map)
        out_img=self.conv1(out_img)
       
        out_img=self.Batch_Norm_1(out_img)
      #   out_img=F.leaky_relu(out_img)
        out_img=self.deconv2(out_img)
        out_img=self.conv2(out_img)
        
        out_img=self.Batch_Norm_2(out_img)
      #   out_img=F.leaky_relu(out_img)
        out_img=self.deconv3(out_img)
        out_img=self.conv3(out_img)
       
        out_img=self.Batch_Norm_3(out_img)
      #   out_img=F.leaky_relu(out_img)
        out_img=self.deconv4(out_img)
        
        out_img=self.conv4(out_img)     
        
        out_img=self.Batch_Norm_4(out_img)
      #   out_img=F.leaky_relu(out_img)
        if not self.MAE_Dec:
            out_img=self.final_classifyer(out_img)
        else:
            out_img=self.final_MAE_head(out_img)
        if y is not None:
              y_seq=self.Fixed_large_model.patchify(y[:,None,None])
              if id_restore_t is not None:
                     y_img=self.Fixed_large_model.unpatchify(y_seq.clone(),Target=True).squeeze(1).squeeze(1)
              else:
                    y_img=self.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
              for cl in torch.unique(y_img):
                if cl==-100:
                       continue
                num_samples_per_class=(y_img==cl).sum()
                To_be_subtracted=torch.zeros((out_img.shape[1],),device=x.device)
                To_be_subtracted[cl.to(torch.long)-1]=1/torch.pow(num_samples_per_class,1/4)
                
                out_img=out_img.permute(1,0,2,3)
                out_img[:,y_img==cl]=(out_img[:,y_img==cl].T-To_be_subtracted).T
                out_img=out_img.permute(1,0,2,3)
        out3=out_img
        if self.MAE_Dec:
             out3=self.Fixed_large_model.patchify(out3[:,:,None])
        
                
        return out3
 
class Model(nn.Module):
    def __init__(self, input_shape,T_input_shape,num_cl, Adapter_depth=1,Seg_Adapter_depth=1):
        super(Model, self).__init__()
        s=18
        kernal_size=input_shape[1]-s*6+1
        kernal_size_T=T_input_shape[1]-s*6+1
        kernal_size_T=5
        self.conv1 = nn.Conv2d(input_shape[1], 6, kernel_size=3, padding='same' ).requires_grad_(requires_grad=True)
        self.conv1_T = nn.Conv3d(1, 1, kernel_size=(kernal_size_T,1,1), stride=(s,1,1) ).requires_grad_(requires_grad=True)
        self.Batch1=nn.BatchNorm2d(6).requires_grad_(requires_grad=True)
        self.Batch1_T=nn.BatchNorm2d(6).requires_grad_(requires_grad=True)
       
        self.Linear_conv=nn.Conv2d(6, 6, kernel_size=1, padding='same')
        model_args_tuple=load_pre_trained_model(input_shape,T_input_shape,random_initi=False)
        self.Fixed_large_model = model_args_tuple[0].requires_grad_(requires_grad=True)
        self.Fixed_large_model_args = model_args_tuple[1]
        # self.Liner_to_Encoder_Adapter= nn.Linear(self.Fixed_large_model_args['embed_dim'],16)
        self.Encoder_Adapter=nn.ModuleList([
            Block(self.Fixed_large_model_args['embed_dim'],self.Fixed_large_model_args['num_heads'],4, qkv_bias=True, norm_layer=nn.LayerNorm,act_layer=nn.LeakyReLU)
            for i in range(Adapter_depth)]).requires_grad_(requires_grad=True)
        self.Encoder_adapter_layer_norm=nn.LayerNorm(self.Fixed_large_model_args['embed_dim'])
        self.decoder_embed=nn.Linear(self.Fixed_large_model_args['embed_dim'],self.Fixed_large_model_args['decoder_embed_dim'])
        # self.Encoder_Adapter=nn.ModuleList([
        #     Block(16,1,4, qkv_bias=True, norm_layer=nn.LayerNorm,act_layer=nn.LeakyReLU)
        #     for i in range(Adapter_depth)]).requires_grad_(requires_grad=True)-----------------------------------
      
        # self.Encoder_Adapter=nn.ModuleList([
        #     Block(self.Fixed_large_model_args['embed_dim'],1,4, qkv_bias=True, norm_layer=nn.LayerNorm,act_layer=nn.LeakyReLU)
        #     for i in range(Adapter_depth)]).requires_grad_(requires_grad=True)
        self.Decoder_Adapter=nn.ModuleList([ Block(self.Fixed_large_model_args['decoder_embed_dim'],self.Fixed_large_model_args['decoder_num_heads'],4, qkv_bias=True, act_layer=nn.LeakyReLU,norm_layer=nn.LayerNorm)
            for i in range(Adapter_depth)]).requires_grad_(requires_grad=True)
        self.Decoder_adapter_layer_norm=nn.LayerNorm(self.Fixed_large_model_args['decoder_embed_dim']).requires_grad_(requires_grad=True)
        # self.Decoder_Adapter=nn.ModuleList([ Block(self.Fixed_large_model_args['decoder_embed_dim'],1,4, qkv_bias=True, act_layer=nn.LeakyReLU,norm_layer=nn.LayerNorm)
        #     for i in range(Adapter_depth)]).requires_grad_(requires_grad=True)
        # self.Seg_Dec= Seg_Dec(input_shape,T_input_shape,num_cl, Adapter_depth=Seg_Adapter_depth)
        # self.Seg_Dec=Seg_Conv((0,self.Fixed_large_model_args['decoder_embed_dim']),input_shape,T_input_shape,num_cl)
        # self.decoder_to_deconv_lin=nn.Linear(self.Fixed_large_model_args['decoder_embed_dim'],16)
        # self.decoder_to_deconv_layer_norm= nn.LayerNorm(16).requires_grad_(requires_grad=True)
        # self.deconv_layer=Seg_Conv_Blender((0,self.Fixed_large_model_args['decoder_embed_dim']),input_shape,T_input_shape,T_input_shape[1])
        self.Dual_Dec=Seg_Conv_Blender((0,self.Fixed_large_model_args['decoder_embed_dim']),input_shape,T_input_shape,num_cl)
        # self.Linear_deconv=nn.Linear(self.Fixed_large_model.patch_embed.patch_size[0]**2*6,self.Fixed_large_model.patch_embed.patch_size[0]**2*T_input_shape[1]).requires_grad_(requires_grad=True)
        # self.Linear_deconv=nn.ConvTranspose3d(1,1,kernel_size=(input_shape[1]-5,1,1), stride=(1,1,1),output_padding=0,padding=0).requires_grad_(requires_grad=True)
        # self.final_layer_norm=nn.LayerNorm(self.Fixed_large_model.patch_embed.patch_size[0]**2*T_input_shape[1]).requires_grad_(requires_grad=True)
      
   
    def forward_layer_1(self, x,Target=False):
        if not Target:
            x = self.conv1(x)
            x = self.Batch1(x)
        else:
            x = self.conv1_T(x[:,None]).squeeze(1)
            x = self.Batch1_T(x)
        
       
       
        x = F.leaky_relu(x)
        # rec = self.Linear_conv(x)
      
        return x
   
    def forward_layer_last(self, x):
        out=self.Fixed_large_model.unpatchify(x.clone()[:,1:]).permute(0,2,1,3,4)
        res= self.Linear_deconv(out)
        res=self.Fixed_large_model.patchify(res.permute(0,2,1,3,4))
        # res=torch.cat([x[:,0][None],res],1)
        res = self.Batch2(res.permute(0,2,1)).permute(0,2,1)
      
        return res
    def forward_layer_last_Lin(self, x,id_restore_s):
       
        res= self.Dual_Dec(x,id_restore_s=id_restore_s,MAE=True)
        # res=self.final_layer_norm(res)
        # res=torch.cat([x[:,0][None],res],1)
        # res = self.Batch2(res.permute(0,2,1)).permute(0,2,1)
      
        return res


   
    def forward(self, x_s=None,x_t=None, mask_ratio_s=0.1,mask_ratio_t=0.1,y=None,Seg_output=True):
        # with torch.no_grad():
        if not x_s==None:
            x_s=self.forward_layer_1(x_s)[:,:,None]
        if x_t is not None:
            x_t=self.forward_layer_1(x_t,Target=False)[:,:,None]
        with torch.no_grad():

            out1, mask_s,mask_t, ids_restore_s,ids_restore_t=self.Fixed_large_model.forward_encoder(x_s,x_t, mask_ratio_s=mask_ratio_s,mask_ratio_t=mask_ratio_t)
            out1=self.Fixed_large_model.norm(out1)
            # out1=self.Fixed_large_model.forward_decoder_no_pred(out1,ids_restore_s,ids_restore_t)
        
        # if self.mask_check is None:
        #     self.mask_check=mask
        # elif torch.all(self.mask_check==mask):
        #     mask=mask[:,[1,0]]
        #     self.mask_check=mask
        # else:
        #     self.mask_check=mask

        # print(mask)--------------------------------------------
        # out1=self.Liner_to_Encoder_Adapter(out1)
        for i in range(len(self.Encoder_Adapter)):
            out1=self.Encoder_Adapter[i](out1)
        out1=self.Encoder_adapter_layer_norm(out1)

       
        if not Seg_output:
            out3=self.decoder_embed(out1)
            out3=self.Fixed_large_model.forward_decoder_no_pred(out3,ids_restore_s,ids_restore_t)
            # out3=self.Fixed_large_model.decoder_norm(out3)
            # out3=self.Fixed_large_model.decoder_pred(out3)
            for i in range(len(self.Decoder_Adapter)):
                    out3=self.Decoder_Adapter[i](out3)
            out3=self.Decoder_adapter_layer_norm(out3)    
            # out3=self.Fixed_large_model.decoder_pred(out3)    




# out3=self.Fixed_large_model.forward_decoder_no_pred(out1,ids_restore_s,ids_restore_t)
            # out3=self.Fixed_large_model.decoder_norm(out3)
            # for i in range(len(self.Decoder_Adapter)):
            #     out3=self.Decoder_Adapter[i](out3)
            
            # out3=self.Fixed_large_model.decoder_pred(out3)
            # out3=F.sigmoid(self.Batch3(out3.permute(0,2,1)).permute(0,2,1)) 
        
            # out3=self.decoder_to_deconv_lin(out3)
            # out3=self.decoder_to_deconv_layer_norm(out3)
            # out3=self.forward_layer_last(out3)
            out3=self.forward_layer_last_Lin(out3, ids_restore_s)
            out3=F.sigmoid(out3)[:,:]
            
            return out3,mask_s,mask_t
        else:
            if not y== None:
                out3=self.Seg_Dec(out1,ids_restore_s,ids_restore_t,y)
            else:
                out3=self.decoder_embed(out1)
                out3=self.Fixed_large_model.forward_decoder_no_pred(out3,ids_restore_s,ids_restore_t)
            # out3=self.Fixed_large_model.decoder_norm(out3)
            # out3=self.Fixed_large_model.decoder_pred(out3)
                for i in range(len(self.Decoder_Adapter)):
                    out3=self.Decoder_Adapter[i](out3)
                out3=self.Decoder_adapter_layer_norm(out3) 
            # # out3=self.Fixed_large_model.decoder_pred(out3)
                # for i in range(len(self.Decoder_Adapter)):
                #     out3=self.Decoder_Adapter[i](out3)
               
                # out1= self.Decoder_adapter_layer_norm(out1) 
              
                # out3=self.decoder_to_deconv_layer_norm(out1)
                out3=self.Dual_Dec(out3,ids_restore_s,ids_restore_t,MAE=False)
            return out3 
#%% lightning_Method
class lightning_Method(L.LightningModule):
    def __init__(self, S_cnn,num_cl,LR=None) -> None:
        super().__init__()
       
        self.source_model=S_cnn
        self.method=None
        self.automatic_optimization = False
        self.phat=torch.ones((num_cl,))/num_cl
        self.num_cl=num_cl
        self.LR=LR
        
    def training_step(self, batch, batch_idx): 
      
      
        Optimizer=self.optimizers()
        sum_loss=0
        for i in range(1):
            x_img_T,y_tt=batch[0]
            x_img,y=batch[1]
            if self.current_epoch==0:
                y_t=y_tt.clone()
                y_t=y_t.to(torch.long)
                y_t[y_t!=-100]=y_t[y_t!=-100]-1
                y_seq=self.source_model.Fixed_large_model.patchify(y_t[:,None,None])
                y_t=self.source_model.Fixed_large_model.unpatchify(y_seq.clone(),Target=True).squeeze(1).squeeze(1)
                mean_loss=[]

            

                model_out=self.source_model.eval()(x_t=x_img_T.clone(),mask_ratio_t=0,Seg_output=True)
                losss=F.cross_entropy(model_out,y_t)
                out=torch.max(model_out, 1)[1].squeeze()
        

                y_t=torch.squeeze(y_t)
                out=out[y_t!=-100]
                y_t=y_t[y_t!=-100]

                mf1=f1_score(y_t.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro')
                mIoU= jaccard_score(y_t.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro')
                
                mean_loss.append(losss.item())
                losss=torch.tensor(mean_loss).mean()
                self.trainer.logger.experiment.add_scalar('Source val loss',losss,self.current_epoch-1)
                self.trainer.logger.experiment.add_scalar('mF1',torch.tensor(mf1),self.current_epoch-1)
                self.trainer.logger.experiment.add_scalar('mIoU',torch.tensor(mIoU),self.current_epoch-1)

            #######################################Monitor Gradient############################################################
            model_out=self.source_model.eval()(x_s=x_img,x_t=x_img_T,mask_ratio_s=0,mask_ratio_t=.99,Seg_output=False)
            x_pre_seq=model_out[0]
            mask=model_out[2]
            # if len(x_img_T.shape)==4:
            #         x_img_T=x_img_T[:,:,None]
            
            loss=self.source_model.Fixed_large_model.forward_loss( x_img_T[:,:,None],  x_pre_seq, mask)
            sum_loss=sum_loss+loss
            loss_MAE=1*sum_loss/1
            Optimizer.zero_grad()
            self.manual_backward(1*loss_MAE)
            self.plot_weights_mean_variance(self.source_model,loss_type='MAE')
            
            model_output=self.source_model.eval()(x_s=x_img, mask_ratio_s=0,Seg_output=True)
            y[y!=-100]=(y[y!=-100]-1)
            y=y.to(torch.long)
            y_seq=self.source_model.Fixed_large_model.patchify(y[:,None,None])
            y=self.source_model.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
            loss_SEG=F.cross_entropy( model_output,y)
            Optimizer.zero_grad()
            self.manual_backward(loss_SEG)
            self.plot_weights_mean_variance(self.source_model,loss_type='Seg')
            # x_img,x_img_T=batch
            # self.source_model.Fixed_large_model.requires_grad_(requires_grad=False)
            
            
            
            #####################################################################################################################
            sum_loss=0
            y[y!=-100]=(y[y!=-100]+1)
            # x_img_T_T= x_img_T.clone()
            # x_img_T_T[:,:,y_tt.squeeze()==3]=x_img_T_T[:,:,y_tt.squeeze()==3]+ 100*torch.torch.randn( x_img_T_T[:,:,y_tt.squeeze()==3].size(), dtype= x_img_T_T[:,:,y_tt.squeeze()==3].dtype, layout= x_img_T_T[:,:,y_tt.squeeze()==3].layout, device=self.device,generator=torch.Generator(device=self.device).manual_seed(50))
            model_out=self.source_model.train()(x_s=x_img,x_t=x_img_T,mask_ratio_s=0,mask_ratio_t=.99,Seg_output=False)
            x_pre_seq=model_out[0]
            mask=model_out[2]
        #     # if len(x_img_T.shape)==4:
        #             # x_img_T=x_img_T[:,:,None]
            
            loss=self.source_model.Fixed_large_model.forward_loss( x_img_T[:,:,None],  x_pre_seq, mask)
            sum_loss=sum_loss+loss
        loss_MAE=1*sum_loss/1
        # loss_MAE=0.0
        Optimizer.zero_grad()
        self.manual_backward(1*loss_MAE)
        
        model_output=self.source_model.train()(x_s=x_img, mask_ratio_s=0,Seg_output=True)
        y[y!=-100]=(y[y!=-100]-1)
        y=y.to(torch.long)
        y_seq=self.source_model.Fixed_large_model.patchify(y[:,None,None])
        y=self.source_model.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
        loss_SEG=F.cross_entropy( model_output,y)
        self.manual_backward(1*loss_SEG)
        # loss_SEG=0.0
        ##################################################### Debaised ############################################################3
        

        ##############################################Few labeled samples#############################################3
        model_out=self.source_model.eval()(x_t=x_img_T.clone(),mask_ratio_t=0,Seg_output=True)
        y_tt[y_tt!=-100]=(y_tt[y_tt!=-100]-1)
        y_tt=y_tt.to(torch.long)
        y_seq=self.source_model.Fixed_large_model.patchify(y_tt[:,None,None])
        y_tt=self.source_model.Fixed_large_model.unpatchify(y_seq.clone(),Target=True).squeeze(1).squeeze(1)
        Target_Seg_labeled_loss=F.cross_entropy( model_out,y_tt)
        self.manual_backward(1*Target_Seg_labeled_loss)
        # Target_Seg_labeled_loss=0.0

        ###################################################Entropy loss#####################################################
        o_t=self.source_model.train()(x_t=x_img_T.clone(),mask_ratio_t=0,Seg_output=True)
        o_t=F.softmax(o_t,dim=1)
        loss_entropy= self.entropy_loss(o_t)
        self.manual_backward(loss_entropy)
        # loss_entropy=0.0


        
        
        Optimizer.step()
        loss=loss_MAE+1*loss_SEG+1*Target_Seg_labeled_loss+loss_entropy


        self.log('Source Total train loss',loss,on_epoch=True,prog_bar=True)
        self.log('Source MAE train loss',loss_MAE,on_epoch=True,prog_bar=True)
        self.log('Source SEG train loss',loss_SEG,on_epoch=True,prog_bar=True)
        self.log('loss_entropy',loss_entropy,on_epoch=True,prog_bar=True)
        self.log('Target_Seg_labeled_loss',Target_Seg_labeled_loss,on_epoch=True,prog_bar=True)
        Optimizer.zero_grad()
        
        
        
        
        return loss
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            for i in range(1):
                x_img_T,y=batch
                y=y.to(torch.long)
                y[y!=-100]=y[y!=-100]-1
                y_seq=self.source_model.Fixed_large_model.patchify(y[:,None,None])
                y=self.source_model.Fixed_large_model.unpatchify(y_seq.clone(),Target=True).squeeze(1).squeeze(1)
                mean_loss=[]

            

                model_out=self.source_model.eval()(x_t=x_img_T.clone(),mask_ratio_t=0,Seg_output=True)
                # model_out=self.source_model.train()(x_s=x_img_T, mask_ratio_s=0,Seg_output=True)
               

                
                losss=F.cross_entropy(model_out,y)
                out=torch.max(model_out, 1)[1].squeeze()
        
    
                y=torch.squeeze(y)
                out=out[y!=-100]
                y=y[y!=-100]

                mf1=f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro')
                mIoU= jaccard_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(),average='macro')
                
                mean_loss.append(losss.item())
            losss=torch.tensor(mean_loss).mean()
            self.log('Source val loss',losss,on_epoch=True,prog_bar=True)
            # self.trainer.logger.experiment.add_scalar('Source val loss',losss,self.current_epoch)
            # self.trainer.logger.experiment.add_scalar('mF1',torch.tensor(mf1),self.current_epoch)
            self.log('mF1',torch.tensor(mf1),on_epoch=True,prog_bar=True)
            self.trainer.logger.experiment.add_scalar('mIoU',torch.tensor(mIoU),self.current_epoch)
        return losss
          
     
        
    
    
    def configure_optimizers(self):
        params_to_optimize = []

# Iterate over all named parameters in the model
        for name, param in self.source_model.named_parameters():
        # Check if the parameter is not part of Fixed_large_model
            if  name.startswith('Fixed_large_model'):
                for names,param in self.source_model.Fixed_large_model.named_parameters():
                    if  not names.startswith (('patch_embed','pos_embed','norm','cls_token','blocks')):
                        params_to_optimize.append(param)
                        param.requires_grad_(requires_grad=False)
                        # pass
                    else: 
                        param.requires_grad_(requires_grad=False)
            
            elif name.startswith('Seg_Dec'):
                     param.requires_grad_(requires_grad=True)            # pass
            # elif 'Linear_deconv' in name or 'final_layer_norm' in name :            
            #         param.requires_grad_(requires_grad=False)
            else:        
                    params_to_optimize.append(param)
                    param.requires_grad_(requires_grad=True)
               

        return torch.optim.AdamW( self.source_model.parameters(), lr= self.LR)
    @torch.no_grad()
    def plot_weights_mean_variance(self,model,loss_type='MAE'):
      if loss_type=='D_cnn':
        model=self.D_cnn
      else:
        pass
    # Iterate over each layer in the model
      All_weight_grad=[]
      for name, param in model.named_parameters():
          if 'weight' in name and ('Encoder_Adapter' in name or 'Decoder_Adapter' in name ):
              # Create a new figure for each layer

              name_parts = name.split(".")
              tensor_board_tag_weights= os.path.join(loss_type,*name_parts)
              name_parts[-1] = "grad"
              tensor_board_tag_grad= os.path.join(loss_type,*name_parts)
              name_parts[-1] = "norm_2"
              tensor_board_tag_norm2= os.path.join(loss_type,*name_parts)
              name_parts[-1] = "Big_norm"
              tensor_board_tag_Big_norm= os.path.join(loss_type,"Big_norm")
                      

              weights = param.data.flatten()
              if param.grad is not None:
                gradients = param.grad.data.flatten()
                All_weight_grad.append(gradients)
                grad_norm2=torch.linalg.vector_norm(gradients)
                grad_mean=gradients.mean()
                grad_var=gradients.var()
                self.trainer.logger.experiment.add_scalars(tensor_board_tag_grad, {'G_Mean':grad_mean,'G_Variance':grad_var}, self.current_epoch)
                self.trainer.logger.experiment.add_scalar(tensor_board_tag_norm2, grad_norm2 , self.current_epoch)
                

              # Calculate mean and variance
            
              mean = weights.mean()
              variance = weights.var()
              
              self.trainer.logger.experiment.add_scalars( tensor_board_tag_weights, {'W_Mean':mean,'W_Variance':variance}, self.current_epoch)
      if len(All_weight_grad)!=0:
        All_weight_grad=torch.cat(All_weight_grad)
        Big_norm=torch.linalg.vector_norm(All_weight_grad)
        self.trainer.logger.experiment.add_scalar(tensor_board_tag_Big_norm,  Big_norm , self.current_epoch)

    def entropy_loss(self,v):
    
        assert v.dim() == 4
        n, c, h, w = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
#%% DATrain
#DATrain
def DATrain(Dataset,Train=True, batch_size=4,num_workers=16,Result_path=None,random_seed=None,lightgin_log_path=None,keep_train=False,Num_of_Samples=None,Exp_name=None,EPOCH=None,LR=None,devices=None,accelerator=None):
    DA_Datamodule=Dataset.DA_Datamodule(batch_size=batch_size,num_workers=num_workers,Num_of_Samples=Num_of_Samples)
   
   
   
        





    
    if False:
        Checkpoint_callback=ModelCheckpoint(dirpath=Result_path,filename='S_best_model',monitor='Source val loss')
        list_of_files = glob.glob(os.path.join(Result_path, '*.ckpt')) 
        latest_ckpt_path_string= max(list_of_files, key=os.path.getctime)
        return DA_Datamodule,  latest_ckpt_path_string
    else:
        if random_seed is not None:
            seed_everything(random_seed,workers=True)
       
        Checkpoint_callback=ModelCheckpoint(dirpath=Result_path,filename='best_model_mF1-{epoch:03d}-{mF1:.6f}',monitor='mF1',mode='max')
        
        T_batch=DA_Datamodule.get_batch()
        S_batch=DA_Datamodule.get_batch(batch_Datset=' ')
        #####################keep training ##########################################
        if keep_train:
            Tensorborad_logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(name=Exp_name,save_dir=lightgin_log_path,version=0)
            
            
            
            ##########################Get the latest file ##########################################
            latest_ckpt_path_string= ' '
            ##########################Get by the epoch number ##########################################
           
            ################################################################################################
            Source_model=Model(S_batch[0].shape,T_batch[0].shape,DA_Datamodule.cl_num,Adapter_depth=1)
            light_source_model=  lightning_Method.load_from_checkpoint(latest_ckpt_path_string,  S_cnn= Source_model,num_cl=DA_Datamodule.cl_num)
            trainer = L.Trainer(max_epochs=2000,logger=Tensorborad_logger,devices=devices,callbacks=[Checkpoint_callback],deterministic='warn',benchmark=False)
            trainer.fit(model=light_source_model, datamodule=DA_Datamodule,ckpt_path=latest_ckpt_path_string)
            
            return  light_source_model.source_model,DA_Datamodule, light_source_model, Checkpoint_callback.best_model_path



        
        
        
        
        
        
        Source_model=Model(S_batch[0].shape,T_batch[0].shape,DA_Datamodule.cl_num,Adapter_depth=1)

      
        
        light_source_model= lightning_Method(Source_model, DA_Datamodule.cl_num,LR=LR)
       
        Tensorborad_logger=L.pytorch.loggers.tensorboard.TensorBoardLogger(name=Exp_name,save_dir=lightgin_log_path)
        
        if False:
            if accelerator_type=="gpu":
       
                trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator_type,max_epochs=EPOCH,callbacks=[Checkpoint_callback], devices=devices,deterministic='warn',benchmark=False,strategy="ddp_notebook_find_unused_parameters_true")
            else:
                trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator_type,max_epochs=EPOCH,deterministic='warn',benchmark=False,strategy='ddp_cpu')
        else:
            if accelerator=="gpu":
       
                trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator,max_epochs=EPOCH,callbacks=[Checkpoint_callback], devices=devices,deterministic='warn',benchmark=False)
            else:
                trainer = L.Trainer(logger=Tensorborad_logger,accelerator=accelerator,max_epochs=EPOCH,deterministic='warn',benchmark=False)

        
        
        trainer.fit(model=light_source_model, datamodule=DA_Datamodule)
        return light_source_model.source_model,DA_Datamodule, light_source_model, Checkpoint_callback.best_model_path
    
#%% Configiration
cfg={'EPOCH':1000, 'LR':10e-4,'Num_of_Samples':2000}