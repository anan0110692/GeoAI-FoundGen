import copy
import glob
import os

import dill
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataf
import torchgan
import lightning as L
import yaml

from collections import OrderedDict
from functools import partial
from matplotlib.colors import ListedColormap, hex2color
from scipy import io as sio
from scipy import signal
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from einops import rearrange
from prithvi.Prithvi import MaskedAutoencoderViT


Dataset_name = 'C2Seg_AB'


def mapping_mask(mask):
    mapped_mask = np.ones_like(mask) * -100
    unique_classes = np.unique(mask)
    mapped_classes = np.zeros((unique_classes.max().astype(np.int64) + 1,))
    mapped_classes[unique_classes[unique_classes != -100].astype(np.int64)] = np.arange(
        1, unique_classes[unique_classes != -100].size + 1
    )
    mapped_mask[mask != -100] = mapped_classes[mask[mask != -100].astype(np.int64)]
    return mapped_mask


def select_well_represented_common_classes(split=True):
    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Datafiles', 'C2Seg_AB', 'berlin_multimodal.mat'
    )
    source_data = sio.loadmat(source_path)
    source_cube = source_data['HSI'].astype(np.float32)
    source_label = source_data['label'].astype(np.float32)
    source_data_dic = {'raw': source_cube, 'label': source_label}

    target_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'Datafiles', 'C2Seg_AB', 'augsburg_multimodal.mat'
    )
    target_data = sio.loadmat(target_path)
    target_cube = target_data['HSI'].astype(np.float32)
    target_label = target_data['label'].astype(np.float32)
    target_data_dic = {'raw': target_cube, 'label': target_label}

    source_unique_classes, source_histo = np.unique(source_data_dic['label'], return_counts=True)
    source_mask = source_data_dic['label'].astype(np.int64)
    target_mask = target_data_dic['label'].astype(np.int64)
    target_unique_classes, _ = np.unique(target_data_dic['label'], return_counts=True)
    common_classes, s_common_idx, t_common_idx = np.intersect1d(
        source_unique_classes, target_unique_classes, assume_unique=True, return_indices=True
    )
    common_from_s_histo = source_histo[s_common_idx]
    common_from_s = source_unique_classes[s_common_idx]
    underrepresented_classes = common_from_s[common_from_s_histo < np.quantile(common_from_s_histo, .3)]
    if not split:
        source_mask[np.isin(source_mask.copy(), underrepresented_classes)] = -100
        target_mask[np.isin(target_mask.copy(), underrepresented_classes)] = -100
        return source_data_dic['Mosaic_image'], source_mask, target_data_dic['Mosaic_image'], target_mask
    else:
        common_classes, _, _ = np.intersect1d(
            source_unique_classes, target_unique_classes, assume_unique=True, return_indices=True
        )
        return (
            source_data_dic['raw'],
            source_data_dic['label'],
            target_data_dic['raw'],
            target_data_dic['label'],
            common_classes.size,
            None,
        )


def clnum():
    return select_well_represented_common_classes(split=True)[4]


def source_data_generator(batch_size=1, num_workers=16):
    source_dataset = select_well_represented_common_classes(split=True)
    hu_cube = source_dataset[0][0:1000, :500, 2:].copy().astype(np.float32)
    hu_cube_t = source_dataset[2][:, :].copy().astype(np.float32) / (pow(2, 14) - 1)
    hu_cube = hu_cube[None]
    hu_cube_t = hu_cube_t[None]
    hu_cube = (hu_cube - hu_cube.min()) / (hu_cube.max() - hu_cube.min())
    hu_cube_t = (hu_cube_t - hu_cube_t.min()) / (hu_cube_t.max() - hu_cube_t.min())
    hu_cube = torch.from_numpy(hu_cube[:, :, :, :]).permute((0, 3, 1, 2))
    hu_cube_t = torch.from_numpy(hu_cube_t[:, :, :, :]).permute((0, 3, 1, 2))
    ch_num = hu_cube.shape[1]
    ch_num_t = hu_cube_t.shape[1]
    tr_label = source_dataset[1][:1000, :500]

    tr_label = tr_label.astype(np.float32)
    tr_label[tr_label == 0] = -100
    original_unique_classes = np.unique(tr_label)
    tr_label = mapping_mask(tr_label)
    tr_label = np.squeeze(tr_label)

    all_train_labels_idx = np.array([])
    all_val_labels_idx = np.array([])
    classes = np.unique(tr_label.flatten())
    classes = classes[classes != -100]
    for i in classes:
        class_indices = np.argwhere(tr_label == i)
        class_train_idx, class_val_idx, _, _ = train_test_split(
            class_indices, np.zeros((class_indices.shape[0],)), train_size=.8, random_state=42
        )
        if all_train_labels_idx.size == 0:
            all_train_labels_idx = class_train_idx
        else:
            all_train_labels_idx = np.concatenate((all_train_labels_idx, class_train_idx))

        if all_val_labels_idx.size == 0:
            all_val_labels_idx = class_val_idx
        else:
            all_val_labels_idx = np.concatenate((all_val_labels_idx, class_val_idx))

    tr_labell = np.ones_like(tr_label) * -100
    va_label = np.ones_like(tr_label) * -100
    va_label[all_val_labels_idx[:, 0], all_val_labels_idx[:, 1]] = tr_label[
        all_val_labels_idx[:, 0], all_val_labels_idx[:, 1]
    ]
    tr_labell[all_train_labels_idx[:, 0], all_train_labels_idx[:, 1]] = tr_label[
        all_train_labels_idx[:, 0], all_train_labels_idx[:, 1]
    ]
    tr_labell = mapping_mask(tr_labell)
    tr_labell = torch.from_numpy(tr_labell)
    tr_labell = tr_labell[None, :, :]
    va_label = mapping_mask(va_label)
    va_label = torch.from_numpy(va_label)
    va_label = va_label[None, :, :]
    dataset_t = dataf.TensorDataset(hu_cube[:, :], tr_labell)
    train_loader = dataf.DataLoader(dataset_t, batch_size=batch_size, num_workers=num_workers)
    dataset_v = dataf.TensorDataset(hu_cube, va_label)
    validation_loader = dataf.DataLoader(dataset_v, batch_size=batch_size, num_workers=num_workers)
    return (
        train_loader,
        validation_loader,
        original_unique_classes,
        ch_num,
        (source_dataset[0][:100, :100].copy().astype(np.float32) / (pow(2, 14) - 1)).transpose(2, 0, 1),
        ch_num_t,
        tr_label,
    )


def target_data_generator(batch_size=1, num_workers=16, Num_of_Samples=None):
    source_dataset = select_well_represented_common_classes(split=True)

    hu_cube = source_dataset[2][:500, :1000, :].copy().astype(np.float32) / (pow(2, 14) - 1)
    hu_cube = hu_cube[None]
    hu_cube = (hu_cube - hu_cube.min()) / (hu_cube.max() - hu_cube.min())
    ch_num = hu_cube.shape[-1]
    hu_cube = torch.from_numpy(hu_cube[:, :, :, :]).permute((0, 3, 1, 2))

    tr_label = source_dataset[3][:500, :1000]

    tr_label = tr_label.astype(np.float32)
    tr_label[tr_label == 0] = -100
    original_unique_classes = np.unique(tr_label)
    tr_label = mapping_mask(tr_label)
    tr_label = np.squeeze(tr_label)

    target_dataset_seed = 41
    seed_everything(target_dataset_seed, workers=True)
    random_state = check_random_state(target_dataset_seed)

    all_train_labels_unlabeled_idx = np.array([])
    all_val_labels_labeled_idx = np.array([])
    all_test_labels_idx = np.array([])
    classes = np.unique(tr_label.flatten())
    classes = classes[classes != -100]
    for i in classes:
        class_indices = np.argwhere(tr_label == i)
        if Num_of_Samples is not None:
            if i == 8 or i == 12 or i == 13:
                class_train_idx, class_test_idx, _, _ = train_test_split(
                    class_indices, np.zeros((class_indices.shape[0],)), train_size=.7, random_state=42
                )
            else:
                class_train_idx, class_test_idx, _, _ = train_test_split(
                    class_indices, np.zeros((class_indices.shape[0],)), train_size=Num_of_Samples, random_state=42
                )
        else:
            if i == 8 or i == 12 or i == 13:
                class_train_idx, class_test_idx, _, _ = train_test_split(
                    class_indices, np.zeros((class_indices.shape[0],)), train_size=.7, random_state=42
                )
            else:
                class_train_idx, class_test_idx, _, _ = train_test_split(
                    class_indices, np.zeros((class_indices.shape[0],)), train_size=2000, random_state=42
                )

        if i == 8 or i == 12 or i == 13:
            class_test_idx, class_val_labeled_idx, _, _ = train_test_split(
                class_test_idx, np.zeros((class_test_idx.shape[0],)), train_size=2 / 3, random_state=42
            )
        else:
            class_test_idx, class_val_labeled_idx, _, _ = train_test_split(
                class_test_idx, np.zeros((class_test_idx.shape[0],)), train_size=.5, random_state=42
            )

        if all_train_labels_unlabeled_idx.size == 0:
            all_train_labels_unlabeled_idx = class_train_idx
        else:
            all_train_labels_unlabeled_idx = np.concatenate(
                (all_train_labels_unlabeled_idx, class_train_idx)
            )

        if all_val_labels_labeled_idx.size == 0:
            all_val_labels_labeled_idx = class_val_labeled_idx
        else:
            all_val_labels_labeled_idx = np.concatenate(
                (all_val_labels_labeled_idx, class_val_labeled_idx)
            )

        if all_test_labels_idx.size == 0:
            all_test_labels_idx = class_test_idx
        else:
            all_test_labels_idx = np.concatenate((all_test_labels_idx, class_test_idx))

    te_label = np.ones_like(tr_label) * -100
    tr_labell = np.ones_like(tr_label) * -100
    va_label_labeled = np.ones_like(tr_label) * -100
    te_label[all_test_labels_idx[:, 0], all_test_labels_idx[:, 1]] = tr_label[
        all_test_labels_idx[:, 0], all_test_labels_idx[:, 1]
    ]
    tr_labell[all_train_labels_unlabeled_idx[:, 0], all_train_labels_unlabeled_idx[:, 1]] = tr_label[
        all_train_labels_unlabeled_idx[:, 0], all_train_labels_unlabeled_idx[:, 1]
    ]
    va_label_labeled[all_val_labels_labeled_idx[:, 0], all_val_labels_labeled_idx[:, 1]] = tr_label[
        all_val_labels_labeled_idx[:, 0], all_val_labels_labeled_idx[:, 1]
    ]

    tr_labell = torch.from_numpy(tr_labell)
    tr_labell = tr_labell[None, :, :]
    te_label = torch.from_numpy(te_label)
    te_label = te_label[None, :, :]
    va_label_labeled = torch.from_numpy(va_label_labeled)
    va_label_labeled = va_label_labeled[None, :, :]

    dataset_t = dataf.TensorDataset(hu_cube, tr_labell)
    train_loader = dataf.DataLoader(dataset_t, batch_size=batch_size, num_workers=num_workers)
    dataset_te = dataf.TensorDataset(hu_cube, te_label)
    test_loader = dataf.DataLoader(dataset_te, batch_size=batch_size, num_workers=num_workers)
    dataset_val_labeled = dataf.TensorDataset(hu_cube, va_label_labeled)
    val_labeled_loader = dataf.DataLoader(dataset_val_labeled, batch_size=batch_size, num_workers=num_workers)
    train_unlabeled_loader = train_loader

    return train_unlabeled_loader, test_loader, val_labeled_loader, tr_label


def color_map_generator():
    lut_colors = {
        0: '000000',
        1: '#01fdfd',
        2: '#fcfbfb',
        3: '#fc0101',
        4: '#dda0dc',
        5: '#8f04cc',
        6: '#ff83fe',
        7: '#ffdd83',
        8: '#ca8540',
        9: '#bdb76b',
        10: '#01fc01',
        11: '#9acc33',
        12: '#8a4413',
        13: '#826ffd',
    }
    color_map_array = [list(int(i * 255) for i in hex2color(v)) for k, v in lut_colors.items()]
    color_map_array = np.array(color_map_array)
    uh_map = ListedColormap(color_map_array.astype(np.float32) / 256.0)
    return uh_map


class Source_Datamodule(L.LightningDataModule):

    def __init__(self, batch_size=3, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        output_tuple = source_data_generator(self.batch_size, self.num_workers)
        self.train_loader = output_tuple[0]
        self.val_loader = output_tuple[1]
        self.original_unique_classes = output_tuple[2]
        self.original_imge = output_tuple[-1]
        self.classes_labels_array = np.array([
            ' ', "Surface water", "Street", "Urban Fabric",
            "Industrial, commercial and transport",
            "Mine, dump, and construction sites",
            "Artificial, vegetated areas", "Arable Land",
            "Permanent Crops", "Pastures", "Forests", "Shrub",
            "Open spaces with no vegetation", "Inland wetlands", 'ignored',
        ])
        self.color_map = color_map_generator()
        self.ch_num = output_tuple[3]
        self.cl_num_T = output_tuple[-2]
        self.cl_num = self.original_unique_classes[self.original_unique_classes != -100].size
        self.Full_GT_mask = output_tuple[-1]

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class Target_Datamodule(L.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=16, Num_of_Samples=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        output_tuple = target_data_generator(self.batch_size, self.num_workers, Num_of_Samples=Num_of_Samples)
        self.train_loader = output_tuple[0]
        self.test_loader = output_tuple[1]
        self.val_loader = output_tuple[-2]
        self.Full_GT_mask = output_tuple[-1]

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def val_dataloader(self):
        return self.val_loader


class DA_Datamodule(L.LightningDataModule):

    def __init__(self, batch_size=1, num_workers=16, Source_Datamodule_arg=None, Num_of_Samples=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if Source_Datamodule_arg is None:
            _source_datamodule = Source_Datamodule(self.batch_size, self.num_workers)
        else:
            _source_datamodule = Source_Datamodule_arg
        _target_datamodule = Target_Datamodule(self.batch_size, self.num_workers, Num_of_Samples=Num_of_Samples)
        self.ch_num = _source_datamodule.ch_num
        self.Source_dataloader = _source_datamodule.train_dataloader()
        self.Target_dataloader = _target_datamodule.train_dataloader()
        self.Target_test_dataloader = _target_datamodule.test_dataloader()
        self.Target_val_dataloader = _target_datamodule.val_dataloader()
        self.classes_labels_array = _source_datamodule.classes_labels_array
        self.color_map = color_map_generator()
        self.Total_class_num = _source_datamodule.classes_labels_array.size
        self.cl_num = _source_datamodule.cl_num
        self.ch_num = _source_datamodule.ch_num
        self.Full_GT_mask = _target_datamodule.Full_GT_mask
        self.original_unique_classes = _source_datamodule.original_unique_classes

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return [self.Target_dataloader, self.Source_dataloader]

    def val_dataloader(self):
        return self.Target_val_dataloader

    def get_batch(self, batch_Datset='Target'):
        da_loader = self.train_dataloader()
        if batch_Datset == 'Target':
            return next(iter(da_loader[0]))
        else:
            return next(iter(da_loader[1]))


def load_pre_trained_model(S_input_shape, T_input_shape, random_initi=False):
    weights_path = os.path.join(os.getcwd(), 'prithvi', 'Prithvi_100M.pt')
    checkpoint = torch.load(weights_path, map_location="cpu")

    model_cfg_path = os.path.join(os.getcwd(), 'prithvi', 'Prithvi_100M_config.yaml')
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]

    model_args["num_frames"] = 1
    model_args["in_chans"] = 6
    model_args["img_size"] = S_input_shape[-2:]
    model_args["T_img_size"] = T_input_shape[-2:]

    model = MaskedAutoencoderViT(**model_args)
    model.eval()

    del checkpoint['pos_embed']
    del checkpoint['decoder_pos_embed']

    if not random_initi:
        _ = model.load_state_dict(checkpoint, strict=False)
    return model, model_args


class Seg_Conv_Blender(nn.Module):

    def __init__(self, dummy_feature_shape, S_input_shape, T_input_shape, num_cl, num_feature=512):
        super(Seg_Conv_Blender, self).__init__()
        temp = load_pre_trained_model(S_input_shape, T_input_shape, random_initi=True)
        self.model_args = temp[-1]
        self.Fixed_large_model = temp[0]
        input_shape = dummy_feature_shape
        belender_seq_length = (
            self.Fixed_large_model.patch_embed.num_patches
            + self.Fixed_large_model.patch_embed_T.num_patches
            + 2
        )
        blender_stride = belender_seq_length // self.Fixed_large_model.patch_embed_T.num_patches
        blender_kernal = belender_seq_length - (self.Fixed_large_model.patch_embed_T.num_patches - 1) * blender_stride

        self.Blender = nn.Conv1d(input_shape[1], input_shape[1], blender_kernal, blender_stride)
        self.deconv1 = nn.ConvTranspose2d(num_feature, num_feature, 2, stride=2)
        self.conv1 = nn.Conv2d(num_feature, num_feature // 2, 2, padding='same')
        self.Batch_Norm_1 = nn.BatchNorm2d(num_feature // 2)
        self.deconv2 = nn.ConvTranspose2d(num_feature // 2, num_feature // 2, 2, stride=2)
        self.conv2 = nn.Conv2d(num_feature // 2, num_feature // 4, 2, padding='same')
        self.Batch_Norm_2 = nn.BatchNorm2d(num_feature // 4)
        self.deconv3 = nn.ConvTranspose2d(num_feature // 4, num_feature // 4, 2, stride=2)
        self.conv3 = nn.Conv2d(num_feature // 4, num_feature // 8, 1, padding='same')
        self.Batch_Norm_3 = nn.BatchNorm2d(num_feature // 8)
        self.deconv4 = nn.ConvTranspose2d(num_feature // 8, num_feature // 8, 2, stride=2)
        self.conv4 = nn.Conv2d(num_feature // 8, num_feature // 16, 3, padding='same')
        self.Batch_Norm_4 = nn.BatchNorm2d(num_feature // 16)
        self.final_classifyer = nn.Conv2d(num_feature // 16, num_cl, 1)
        self.final_MAE_head = nn.Conv2d(num_feature // 16, T_input_shape[1], 1)
        self.MAE_Dec = None

    def forward(self, x_seq, id_restore_s=None, id_restore_t=None, y=None, MAE=False):
        self.MAE_Dec = MAE
        if not self.MAE_Dec:
            if id_restore_t is not None:
                x_feature_map = torch.reshape(
                    x_seq[:, 1:].flatten(),
                    (
                        x_seq.shape[0],
                        self.model_args["T_img_size"][0] // self.model_args["patch_size"],
                        self.model_args["T_img_size"][1] // self.model_args["patch_size"],
                        -1,
                    ),
                ).permute(0, -1, 1, 2)
            else:
                x_feature_map = torch.reshape(
                    x_seq[:, 1:].flatten(),
                    (
                        x_seq.shape[0],
                        self.model_args["img_size"][0] // self.model_args["patch_size"],
                        self.model_args["img_size"][1] // self.model_args["patch_size"],
                        -1,
                    ),
                ).permute(0, -1, 1, 2)
        else:
            random_indices = torch.randperm(x_seq.shape[1], generator=torch.Generator().manual_seed(50))
            x_seq_shuffled = x_seq[:, random_indices]
            belender_out = self.Blender(x_seq_shuffled.permute((0, -1, 1))).permute((0, -1, 1))
            x_feature_map = torch.reshape(
                belender_out.flatten(),
                (
                    x_seq.shape[0],
                    self.model_args["T_img_size"][0] // self.model_args["patch_size"],
                    self.model_args["T_img_size"][1] // self.model_args["patch_size"],
                    -1,
                ),
            ).permute(0, -1, 1, 2)

        out_img = self.deconv1(x_feature_map)
        out_img = self.conv1(out_img)
        out_img = self.Batch_Norm_1(out_img)
        out_img = self.deconv2(out_img)
        out_img = self.conv2(out_img)
        out_img = self.Batch_Norm_2(out_img)
        out_img = self.deconv3(out_img)
        out_img = self.conv3(out_img)
        out_img = self.Batch_Norm_3(out_img)
        out_img = self.deconv4(out_img)
        out_img = self.conv4(out_img)
        out_img = self.Batch_Norm_4(out_img)
        if not self.MAE_Dec:
            out_img = self.final_classifyer(out_img)
        else:
            out_img = self.final_MAE_head(out_img)
        if y is not None:
            y_seq = self.Fixed_large_model.patchify(y[:, None, None])
            if id_restore_t is not None:
                y_img = self.Fixed_large_model.unpatchify(y_seq.clone(), Target=True).squeeze(1).squeeze(1)
            else:
                y_img = self.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
            for cl in torch.unique(y_img):
                if cl == -100:
                    continue
                num_samples_per_class = (y_img == cl).sum()
                to_be_subtracted = torch.zeros((out_img.shape[1],), device=x_seq.device)
                to_be_subtracted[cl.to(torch.long) - 1] = 1 / torch.pow(num_samples_per_class, 1 / 4)
                out_img = out_img.permute(1, 0, 2, 3)
                out_img[:, y_img == cl] = (out_img[:, y_img == cl].T - to_be_subtracted).T
                out_img = out_img.permute(1, 0, 2, 3)
        out3 = out_img
        if self.MAE_Dec:
            out3 = self.Fixed_large_model.patchify(out3[:, :, None])
        return out3


class Model(nn.Module):

    def __init__(self, input_shape, T_input_shape, num_cl, Adapter_depth=1, Seg_Adapter_depth=1):
        super(Model, self).__init__()
        s = 18
        kernal_size_T = 5
        self.conv1 = nn.Conv2d(input_shape[1], 6, kernel_size=3, padding='same').requires_grad_(requires_grad=True)
        self.conv1_T = nn.Conv3d(1, 1, kernel_size=(kernal_size_T, 1, 1), stride=(s, 1, 1)).requires_grad_(
            requires_grad=True
        )
        self.Batch1 = nn.BatchNorm2d(6).requires_grad_(requires_grad=True)
        self.Batch1_T = nn.BatchNorm2d(6).requires_grad_(requires_grad=True)
        self.Linear_conv = nn.Conv2d(6, 6, kernel_size=1, padding='same')
        model_args_tuple = load_pre_trained_model(input_shape, T_input_shape, random_initi=False)
        self.Fixed_large_model = model_args_tuple[0].requires_grad_(requires_grad=True)
        self.Fixed_large_model_args = model_args_tuple[1]
        self.Encoder_Adapter = nn.ModuleList([
            Block(
                self.Fixed_large_model_args['embed_dim'],
                self.Fixed_large_model_args['num_heads'],
                4,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.LeakyReLU,
            )
            for _ in range(Adapter_depth)
        ]).requires_grad_(requires_grad=True)
        self.Encoder_adapter_layer_norm = nn.LayerNorm(self.Fixed_large_model_args['embed_dim'])
        self.decoder_embed = nn.Linear(
            self.Fixed_large_model_args['embed_dim'], self.Fixed_large_model_args['decoder_embed_dim']
        )
        self.Decoder_Adapter = nn.ModuleList([
            Block(
                self.Fixed_large_model_args['decoder_embed_dim'],
                self.Fixed_large_model_args['decoder_num_heads'],
                4,
                qkv_bias=True,
                act_layer=nn.LeakyReLU,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(Adapter_depth)
        ]).requires_grad_(requires_grad=True)
        self.Decoder_adapter_layer_norm = nn.LayerNorm(
            self.Fixed_large_model_args['decoder_embed_dim']
        ).requires_grad_(requires_grad=True)
        self.Dual_Dec = Seg_Conv_Blender(
            (0, self.Fixed_large_model_args['decoder_embed_dim']), input_shape, T_input_shape, num_cl
        )

    def forward_layer_1(self, x, Target=False):
        if not Target:
            x = self.conv1(x)
            x = self.Batch1(x)
        else:
            x = self.conv1_T(x[:, None]).squeeze(1)
            x = self.Batch1_T(x)
        x = F.leaky_relu(x)
        return x

    def forward_layer_last_Lin(self, x, id_restore_s):
        res = self.Dual_Dec(x, id_restore_s=id_restore_s, MAE=True)
        return res

    def forward(self, x_s=None, x_t=None, mask_ratio_s=0.1, mask_ratio_t=0.1, y=None, Seg_output=True):
        if x_s is not None:
            x_s = self.forward_layer_1(x_s)[:, :, None]
        if x_t is not None:
            x_t = self.forward_layer_1(x_t, Target=False)[:, :, None]
        with torch.no_grad():
            out1, mask_s, mask_t, ids_restore_s, ids_restore_t = self.Fixed_large_model.forward_encoder(
                x_s, x_t, mask_ratio_s=mask_ratio_s, mask_ratio_t=mask_ratio_t
            )
            out1 = self.Fixed_large_model.norm(out1)

        for i in range(len(self.Encoder_Adapter)):
            out1 = self.Encoder_Adapter[i](out1)
        out1 = self.Encoder_adapter_layer_norm(out1)

        if not Seg_output:
            out3 = self.decoder_embed(out1)
            out3 = self.Fixed_large_model.forward_decoder_no_pred(out3, ids_restore_s, ids_restore_t)
            for i in range(len(self.Decoder_Adapter)):
                out3 = self.Decoder_Adapter[i](out3)
            out3 = self.Decoder_adapter_layer_norm(out3)
            out3 = self.forward_layer_last_Lin(out3, ids_restore_s)
            out3 = F.sigmoid(out3)[:, :]
            return out3, mask_s, mask_t
        else:
            out3 = self.decoder_embed(out1)
            out3 = self.Fixed_large_model.forward_decoder_no_pred(out3, ids_restore_s, ids_restore_t)
            for i in range(len(self.Decoder_Adapter)):
                out3 = self.Decoder_Adapter[i](out3)
            out3 = self.Decoder_adapter_layer_norm(out3)
            out3 = self.Dual_Dec(out3, ids_restore_s, ids_restore_t, MAE=False)
            return out3


class lightning_Method(L.LightningModule):

    def __init__(self, S_cnn, num_cl, LR=None) -> None:
        super().__init__()
        self.source_model = S_cnn
        self.method = None
        self.automatic_optimization = False
        self.phat = torch.ones((num_cl,)) / num_cl
        self.num_cl = num_cl
        self.LR = LR

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sum_loss = 0
        for i in range(1):
            x_img_T, y_tt = batch[0]
            x_img, y = batch[1]
            if self.current_epoch == 0:
                y_t = y_tt.clone()
                y_t = y_t.to(torch.long)
                y_t[y_t != -100] = y_t[y_t != -100] - 1
                y_seq = self.source_model.Fixed_large_model.patchify(y_t[:, None, None])
                y_t = self.source_model.Fixed_large_model.unpatchify(
                    y_seq.clone(), Target=True
                ).squeeze(1).squeeze(1)
                mean_loss = []

                model_out = self.source_model.eval()(x_t=x_img_T.clone(), mask_ratio_t=0, Seg_output=True)
                losss = F.cross_entropy(model_out, y_t)
                out = torch.max(model_out, 1)[1].squeeze()

                y_t = torch.squeeze(y_t)
                out = out[y_t != -100]
                y_t = y_t[y_t != -100]

                mf1 = f1_score(y_t.cpu().detach().numpy(), out.cpu().detach().numpy(), average='macro')
                miou = jaccard_score(y_t.cpu().detach().numpy(), out.cpu().detach().numpy(), average='macro')

                mean_loss.append(losss.item())
                losss = torch.tensor(mean_loss).mean()
                self.trainer.logger.experiment.add_scalar('Source val loss', losss, self.current_epoch - 1)
                self.trainer.logger.experiment.add_scalar('mF1', torch.tensor(mf1), self.current_epoch - 1)
                self.trainer.logger.experiment.add_scalar('mIoU', torch.tensor(miou), self.current_epoch - 1)

            model_out = self.source_model.eval()(
                x_s=x_img, x_t=x_img_T, mask_ratio_s=0, mask_ratio_t=.99, Seg_output=False
            )
            x_pre_seq = model_out[0]
            mask = model_out[2]
            loss = self.source_model.Fixed_large_model.forward_loss(x_img_T[:, :, None], x_pre_seq, mask)
            sum_loss = sum_loss + loss
            loss_mae = 1 * sum_loss / 1
            optimizer.zero_grad()
            self.manual_backward(1 * loss_mae)
            self.plot_weights_mean_variance(self.source_model, loss_type='MAE')

            model_output = self.source_model.eval()(x_s=x_img, mask_ratio_s=0, Seg_output=True)
            y[y != -100] = y[y != -100] - 1
            y = y.to(torch.long)
            y_seq = self.source_model.Fixed_large_model.patchify(y[:, None, None])
            y = self.source_model.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
            loss_seg = F.cross_entropy(model_output, y)
            optimizer.zero_grad()
            self.manual_backward(loss_seg)
            self.plot_weights_mean_variance(self.source_model, loss_type='Seg')

            sum_loss = 0
            y[y != -100] = y[y != -100] + 1
            model_out = self.source_model.train()(
                x_s=x_img, x_t=x_img_T, mask_ratio_s=0, mask_ratio_t=.99, Seg_output=False
            )
            x_pre_seq = model_out[0]
            mask = model_out[2]
            loss = self.source_model.Fixed_large_model.forward_loss(x_img_T[:, :, None], x_pre_seq, mask)
            sum_loss = sum_loss + loss
        loss_mae = 1 * sum_loss / 1
        optimizer.zero_grad()
        self.manual_backward(1 * loss_mae)

        model_output = self.source_model.train()(x_s=x_img, mask_ratio_s=0, Seg_output=True)
        y[y != -100] = y[y != -100] - 1
        y = y.to(torch.long)
        y_seq = self.source_model.Fixed_large_model.patchify(y[:, None, None])
        y = self.source_model.Fixed_large_model.unpatchify(y_seq.clone()).squeeze(1).squeeze(1)
        loss_seg = F.cross_entropy(model_output, y)
        self.manual_backward(1 * loss_seg)

        model_out = self.source_model.eval()(x_t=x_img_T.clone(), mask_ratio_t=0, Seg_output=True)
        y_tt[y_tt != -100] = y_tt[y_tt != -100] - 1
        y_tt = y_tt.to(torch.long)
        y_seq = self.source_model.Fixed_large_model.patchify(y_tt[:, None, None])
        y_tt = self.source_model.Fixed_large_model.unpatchify(
            y_seq.clone(), Target=True
        ).squeeze(1).squeeze(1)
        target_seg_labeled_loss = F.cross_entropy(model_out, y_tt)
        self.manual_backward(1 * target_seg_labeled_loss)

        o_t = self.source_model.train()(x_t=x_img_T.clone(), mask_ratio_t=0, Seg_output=True)
        o_t = F.softmax(o_t, dim=1)
        loss_entropy = self.entropy_loss(o_t)
        self.manual_backward(loss_entropy)

        optimizer.step()
        loss = loss_mae + 1 * loss_seg + 1 * target_seg_labeled_loss + loss_entropy

        self.log('Source Total train loss', loss, on_epoch=True, prog_bar=True)
        self.log('Source MAE train loss', loss_mae, on_epoch=True, prog_bar=True)
        self.log('Source SEG train loss', loss_seg, on_epoch=True, prog_bar=True)
        self.log('loss_entropy', loss_entropy, on_epoch=True, prog_bar=True)
        self.log('Target_Seg_labeled_loss', target_seg_labeled_loss, on_epoch=True, prog_bar=True)
        optimizer.zero_grad()

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            for i in range(1):
                x_img_T, y = batch
                y = y.to(torch.long)
                y[y != -100] = y[y != -100] - 1
                y_seq = self.source_model.Fixed_large_model.patchify(y[:, None, None])
                y = self.source_model.Fixed_large_model.unpatchify(
                    y_seq.clone(), Target=True
                ).squeeze(1).squeeze(1)
                mean_loss = []

                model_out = self.source_model.eval()(x_t=x_img_T.clone(), mask_ratio_t=0, Seg_output=True)

                losss = F.cross_entropy(model_out, y)
                out = torch.max(model_out, 1)[1].squeeze()

                y = torch.squeeze(y)
                out = out[y != -100]
                y = y[y != -100]

                mf1 = f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(), average='macro')
                miou = jaccard_score(y.cpu().detach().numpy(), out.cpu().detach().numpy(), average='macro')

                mean_loss.append(losss.item())
            losss = torch.tensor(mean_loss).mean()
            self.log('Source val loss', losss, on_epoch=True, prog_bar=True)
            self.log('mF1', torch.tensor(mf1), on_epoch=True, prog_bar=True)
            self.trainer.logger.experiment.add_scalar('mIoU', torch.tensor(miou), self.current_epoch)
        return losss

    def configure_optimizers(self):
        params_to_optimize = []
        for name, param in self.source_model.named_parameters():
            if name.startswith('Fixed_large_model'):
                for names, param in self.source_model.Fixed_large_model.named_parameters():
                    if not names.startswith(('patch_embed', 'pos_embed', 'norm', 'cls_token', 'blocks')):
                        params_to_optimize.append(param)
                        param.requires_grad_(requires_grad=False)
                    else:
                        param.requires_grad_(requires_grad=False)
            elif name.startswith('Seg_Dec'):
                param.requires_grad_(requires_grad=True)
            else:
                params_to_optimize.append(param)
                param.requires_grad_(requires_grad=True)

        return torch.optim.AdamW(self.source_model.parameters(), lr=self.LR)

    @torch.no_grad()
    def plot_weights_mean_variance(self, model, loss_type='MAE'):
        all_weight_grad = []
        for name, param in model.named_parameters():
            if 'weight' in name and ('Encoder_Adapter' in name or 'Decoder_Adapter' in name):
                name_parts = name.split(".")
                tensor_board_tag_weights = os.path.join(loss_type, *name_parts)
                name_parts[-1] = "grad"
                tensor_board_tag_grad = os.path.join(loss_type, *name_parts)
                name_parts[-1] = "norm_2"
                tensor_board_tag_norm2 = os.path.join(loss_type, *name_parts)
                name_parts[-1] = "Big_norm"
                tensor_board_tag_big_norm = os.path.join(loss_type, "Big_norm")

                weights = param.data.flatten()
                if param.grad is not None:
                    gradients = param.grad.data.flatten()
                    all_weight_grad.append(gradients)
                    grad_norm2 = torch.linalg.vector_norm(gradients)
                    grad_mean = gradients.mean()
                    grad_var = gradients.var()
                    self.trainer.logger.experiment.add_scalars(
                        tensor_board_tag_grad,
                        {'G_Mean': grad_mean, 'G_Variance': grad_var},
                        self.current_epoch,
                    )
                    self.trainer.logger.experiment.add_scalar(
                        tensor_board_tag_norm2, grad_norm2, self.current_epoch
                    )

                mean = weights.mean()
                variance = weights.var()
                self.trainer.logger.experiment.add_scalars(
                    tensor_board_tag_weights,
                    {'W_Mean': mean, 'W_Variance': variance},
                    self.current_epoch,
                )
        if len(all_weight_grad) != 0:
            all_weight_grad = torch.cat(all_weight_grad)
            big_norm = torch.linalg.vector_norm(all_weight_grad)
            self.trainer.logger.experiment.add_scalar(tensor_board_tag_big_norm, big_norm, self.current_epoch)

    def entropy_loss(self, v):
        assert v.dim() == 4
        n, c, h, w = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def DATrain(
    Dataset, Train=True, batch_size=4, num_workers=16, Result_path=None, random_seed=None,
    lightgin_log_path=None, keep_train=False, Num_of_Samples=None, Exp_name=None,
    EPOCH=None, LR=None, devices=None, accelerator=None,
):
    da_datamodule = Dataset.DA_Datamodule(
        batch_size=batch_size, num_workers=num_workers, Num_of_Samples=Num_of_Samples
    )

    if False:
        checkpoint_callback = ModelCheckpoint(dirpath=Result_path, filename='S_best_model', monitor='Source val loss')
        list_of_files = glob.glob(os.path.join(Result_path, '*.ckpt'))
        latest_ckpt_path_string = max(list_of_files, key=os.path.getctime)
        return da_datamodule, latest_ckpt_path_string
    else:
        if random_seed is not None:
            seed_everything(random_seed, workers=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=Result_path,
            filename='best_model_mF1-{epoch:03d}-{mF1:.6f}',
            monitor='mF1',
            mode='max',
        )

        t_batch = da_datamodule.get_batch()
        s_batch = da_datamodule.get_batch(batch_Datset=' ')

        if keep_train:
            tensorboard_logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
                name=Exp_name, save_dir=lightgin_log_path, version=0
            )
            latest_ckpt_path_string = ' '
            source_model = Model(s_batch[0].shape, t_batch[0].shape, da_datamodule.cl_num, Adapter_depth=1)
            light_source_model = lightning_Method.load_from_checkpoint(
                latest_ckpt_path_string, S_cnn=source_model, num_cl=da_datamodule.cl_num
            )
            trainer = L.Trainer(
                max_epochs=2000,
                logger=tensorboard_logger,
                devices=devices,
                callbacks=[checkpoint_callback],
                deterministic='warn',
                benchmark=False,
            )
            trainer.fit(model=light_source_model, datamodule=da_datamodule, ckpt_path=latest_ckpt_path_string)
            return light_source_model.source_model, da_datamodule, light_source_model, checkpoint_callback.best_model_path

        source_model = Model(s_batch[0].shape, t_batch[0].shape, da_datamodule.cl_num, Adapter_depth=1)
        light_source_model = lightning_Method(source_model, da_datamodule.cl_num, LR=LR)
        tensorboard_logger = L.pytorch.loggers.tensorboard.TensorBoardLogger(
            name=Exp_name, save_dir=lightgin_log_path
        )

        if accelerator == "gpu":
            trainer = L.Trainer(
                logger=tensorboard_logger,
                accelerator=accelerator,
                max_epochs=EPOCH,
                callbacks=[checkpoint_callback],
                devices=devices,
                deterministic='warn',
                benchmark=False,
            )
        else:
            trainer = L.Trainer(
                logger=tensorboard_logger,
                accelerator=accelerator,
                max_epochs=EPOCH,
                deterministic='warn',
                benchmark=False,
            )

        trainer.fit(model=light_source_model, datamodule=da_datamodule)
        return light_source_model.source_model, da_datamodule, light_source_model, checkpoint_callback.best_model_path


cfg = {'EPOCH': 1000, 'LR': 10e-4, 'Num_of_Samples': 2000}
