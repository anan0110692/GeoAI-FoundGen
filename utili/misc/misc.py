import os
import math

import torch
import torch.nn as nn
import torchgan


def adjust_accelerator_settings(gpu_num=None):
    global devices_s, accelerator_type_s, num_workers_s
    if gpu_num is not None:
        devices_s = [gpu_num]
        num_workers_s = 0
        accelerator_type_s = 'gpu'
        devices_da = [0]
        num_workers_da = 0
        accelerator_type_da = 'gpu'
    else:
        accelerator_type_s = 'cpu'
        accelerator_type_da = 'cpu'
        num_workers_s = 0
        num_workers_da = 0
        devices_s = [0]
        devices_da = [0]
    return devices_s, num_workers_s, accelerator_type_s, devices_da, num_workers_da, accelerator_type_da


def create_result_folder(dataset, exp_name):
    dataset_name = dataset.Dataset_name
    result_folder = 'Results'
    dataset_path = os.path.join(result_folder, dataset_name)
    exp_subfolder = os.path.join(dataset_path, 'Exp_' + exp_name)
    source_subfolder = os.path.join(dataset_path, 'Source', 'Exp_' + exp_name)
    da_subfolder = os.path.join(dataset_path, 'DA', 'Exp_' + exp_name)
    lightning_logs_subfolder = os.path.join(dataset_path, 'lightning_logs')

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(exp_subfolder, exist_ok=True)
    os.makedirs(source_subfolder, exist_ok=True)
    os.makedirs(da_subfolder, exist_ok=True)
    os.makedirs(lightning_logs_subfolder, exist_ok=True)

    return exp_subfolder, lightning_logs_subfolder, dataset_path, source_subfolder, da_subfolder


def xavier_init_weights(model):
    if isinstance(model, nn.Module):
        for name, param in model.named_parameters():
            if ('weight' in name
                    and 'norm' not in name
                    and 'Batch1' not in name
                    and 'Batch2' not in name
                    and 'Batch3' not in name):
                gain = nn.init.calculate_gain('relu') if 'enc' in name else 1.0
                nn.init.xavier_normal_(param.data, gain=gain)


def freeze_final_layers(model):
    for name, mod in model.named_children():
        if name[0] == '_' or name == 'final_layer':
            for param in mod.parameters():
                param.requires_grad = False


def get_trainable_params(model):
    model.requires_grad = True
    freeze_final_layers(model)
    return [p for p in model.parameters() if p.requires_grad]


def _to_pair(val):
    return (val, val) if isinstance(val, int) else val


def _layer_output_shape(sub_model, input_shape):
    if isinstance(sub_model, nn.Sequential):
        last_conv = None
        for layer in sub_model:
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
        return (input_shape[0], last_conv.out_channels, *input_shape[2:])
    elif isinstance(sub_model, nn.MaxPool2d):
        k = _to_pair(sub_model.kernel_size)
        p = _to_pair(sub_model.padding)
        s = _to_pair(sub_model.stride)
        d = _to_pair(sub_model.dilation)
        h = math.floor((input_shape[-2] + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
        w = math.floor((input_shape[-1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
        return (input_shape[0], input_shape[1], h, w)
    elif isinstance(sub_model, nn.ConvTranspose2d):
        k = _to_pair(sub_model.kernel_size)
        p = _to_pair(sub_model.padding)
        s = _to_pair(sub_model.stride)
        d = _to_pair(sub_model.dilation)
        h = (input_shape[-2] - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + 1
        w = (input_shape[-1] - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + 1
        return (input_shape[0], sub_model.out_channels, h, w)
    elif isinstance(sub_model, nn.Conv2d):
        return (input_shape[0], sub_model.out_channels, *input_shape[2:])
    else:
        return input_shape


@torch.no_grad()
def compute_output_padding(model, x):
    upconv_names = [name for name, _ in model.named_children() if 'upconv' in name]
    for name in upconv_names:
        layer = getattr(model, name)
        if isinstance(layer, torchgan.layers.spectralnorm.SpectralNorm2d):
            inner = layer.module
            if not hasattr(inner, 'weight'):
                layer(torch.rand((3, inner.in_channels, 100, 100), device=x.device))
            inner = layer.module
            setattr(inner, 'output_padding', (0, 0))
        else:
            setattr(layer, 'output_padding', (0, 0))

    encoder_shapes = []
    decoder_shapes = []
    size_history = []

    for name, sub_model in model.named_children():
        in_shape = x.shape if not size_history else size_history[-1]

        if not decoder_shapes:
            out_shape = _layer_output_shape(sub_model, in_shape)
            size_history.append(out_shape)
            if 'encoder' in name:
                encoder_shapes.append(out_shape)
            elif 'upconv' in name:
                decoder_shapes.append(out_shape)
        else:
            if 'upconv' in name:
                in_shape = encoder_shapes[-len(decoder_shapes)]
            elif 'final' in name:
                continue
            out_shape = _layer_output_shape(sub_model, in_shape)
            size_history.append(out_shape)
            if 'encoder' in name:
                encoder_shapes.append(out_shape)
            elif 'upconv' in name:
                decoder_shapes.append(out_shape)

    decoder_shapes.reverse()
    output_padding = [
        (int(encoder_shapes[i][-2] - decoder_shapes[i][-2]),
         int(encoder_shapes[i][-1] - decoder_shapes[i][-1]))
        for i in range(len(decoder_shapes))
    ]
    output_padding.reverse()
    return output_padding
