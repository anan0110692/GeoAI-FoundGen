# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
    
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple

import numpy as np

from einops import rearrange

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False,source_grid_size_for_target=None):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """
    if source_grid_size_for_target is None:
        assert embed_dim % 16 == 0

        t_size, h_size, w_size = grid_size

        w_embed_dim = embed_dim // 16 * 6
        h_embed_dim = embed_dim // 16 * 6
        t_embed_dim = embed_dim // 16 * 4

        w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
        h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
        t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

        w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
        h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
        t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

        pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)
    else:
        assert embed_dim % 16 == 0

        t_size, h_size, w_size = grid_size
        t_size_s, h_size_s, w_size_s = source_grid_size_for_target

        w_embed_dim = embed_dim // 16 * 6
        h_embed_dim = embed_dim // 16 * 6
        t_embed_dim = embed_dim // 16 * 4

        w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size_s, w_size_s+w_size))
        h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
        t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

        w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
        h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
        t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

        pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=3,
            tubelet_size=1,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        # self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                              stride=(tubelet_size, patch_size[0], patch_size[1]), bias=bias)
        #-----------------------------------------------------------------------------------------
        Padding_size= self.proj.padding
        if isinstance(Padding_size,int):
                        Padding_size=(torch.tensor(Padding_size),torch.tensor(Padding_size),torch.tensor(Padding_size))
        Stride_size=self.proj.stride
        if isinstance(Stride_size,int):
                        Stride_size=(torch.tensor(Stride_size),torch.tensor(Stride_size),torch.tensor(Stride_size))
        Dilation_size=self.proj.dilation
        if isinstance( Dilation_size,int):
                         Dilation_size=(torch.tensor( Dilation_size),torch.tensor( Dilation_size),torch.tensor( Dilation_size))
        D= ((self.num_frames+2*Padding_size[0]-Dilation_size[0]*(tubelet_size-1)-1)/Stride_size[0]+1)
        H= ((img_size[0]+2*Padding_size[1]-Dilation_size[1]*(patch_size[0]-1)-1)/Stride_size[1]+1)
        W= ((img_size[1]+2*Padding_size[2]-Dilation_size[2]*(patch_size[1]-1)-1)/Stride_size[2]+1)
        D=torch.floor(torch.tensor(D))
        H=torch.floor(torch.tensor(H))
        W=torch.floor(torch.tensor(W))
        self.grid_size=(int(D.item()),int(H.item()),int(W.item()))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, T_img_size=244, img_size=224, patch_size=16,
                 num_frames=3, tubelet_size=1,
                 in_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size,num_frames, tubelet_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.patch_embed_T = PatchEmbed(T_img_size, patch_size,num_frames, tubelet_size, in_chans, embed_dim)
        num_patches_T = self.patch_embed_T.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed_T = nn.Parameter(torch.zeros(1, num_patches_T + 1, embed_dim), requires_grad=False)
        self.pos_embed_T_Seg= nn.Parameter(torch.zeros(1, num_patches_T + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_T = nn.Parameter(torch.zeros(1, num_patches_T + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_T_Seg= nn.Parameter(torch.zeros(1, num_patches_T + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, tubelet_size * patch_size * patch_size * in_chans, bias=True) # decoder to patch
       
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_T = get_3d_sincos_pos_embed(self.pos_embed_T.shape[-1], self.patch_embed_T.grid_size, cls_token=True,source_grid_size_for_target=self.patch_embed.grid_size)
        self.pos_embed_T.data.copy_(torch.from_numpy(pos_embed_T).float().unsqueeze(0))

        pos_embed_T_Seg = get_3d_sincos_pos_embed(self.pos_embed_T_Seg.shape[-1], self.patch_embed_T.grid_size, cls_token=True)
        self.pos_embed_T_Seg.data.copy_(torch.from_numpy(pos_embed_T_Seg).float().unsqueeze(0))


        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_T = get_3d_sincos_pos_embed(self.decoder_pos_embed_T.shape[-1], self.patch_embed_T.grid_size, cls_token=True,source_grid_size_for_target=self.patch_embed.grid_size)
        self.decoder_pos_embed_T.data.copy_(torch.from_numpy(decoder_pos_embed_T).float().unsqueeze(0))

        decoder_pos_embed_T_Seg = get_3d_sincos_pos_embed(self.decoder_pos_embed_T_Seg.shape[-1], self.patch_embed_T.grid_size, cls_token=True)
        self.decoder_pos_embed_T_Seg.data.copy_(torch.from_numpy(decoder_pos_embed_T_Seg).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: B, C, T, H, W
        x: B, L, D
        """
        p = self.patch_embed.patch_size[0]
        tub = self.patch_embed.tubelet_size
        im=imgs.permute(0,2,3,4,1)
        Dimsstride= (
           
            im[0].numel(),
            im.shape[-1]*im.shape[-2]*p,
            im.shape[-1]*p,
            im.shape[-1]*im.shape[-2],
            im.shape[-1],
            1
        )
        x=torch.as_strided_copy(im.flatten(),(im.shape[0],im.shape[2]//p,im.shape[3]//p,p,p,im.shape[-1]),Dimsstride).reshape(im.shape[0],-1,p*p*im.shape[-1])
        # x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)', tub=tub, p=p, q=p,h=imgs.shape[-2], w=imgs.shape[-1])
        

        return x
    ###################################################################Original code######################################
    def unpatchify(self, x,Target=False):
        """
        x: B, L, D
        imgs: B, C, T, H, W
        """
        if not Target:
            p = self.patch_embed.patch_size[0]
            num_p = self.patch_embed.img_size[0] // p
            num_q = self.patch_embed.img_size[1] // p
            tub = self.patch_embed.tubelet_size
            # imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)', h=num_p, w=num_p, tub=tub, p=p, q=p)
            imgs=x.reshape((x.shape[0],self.patch_embed.img_size[0]//p,self.patch_embed.img_size[1]//p,p,p,-1)).permute(0,1,3,2,4,5).reshape(x.shape[0],num_p*p,num_q*p,-1).permute(0,3,1,2)
        else:
            p = self.patch_embed_T.patch_size[0]
            num_p = self.patch_embed_T.img_size[0] // p
            num_q = self.patch_embed_T.img_size[1] // p
            tub = self.patch_embed_T.tubelet_size
            # imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)', h=num_p, w=num_p, tub=tub, p=p, q=p)
            imgs=x.reshape((x.shape[0],self.patch_embed_T.img_size[0]//p,self.patch_embed_T.img_size[1]//p,p,p,-1)).permute(0,1,3,2,4,5).reshape(x.shape[0],num_p*p,num_q*p,-1).permute(0,3,1,2)

        return imgs[:,:,None]
    ##################################################################################################################################
    


    def random_masking(self, x, mask_ratio,typical=True):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device,generator=torch.Generator(device=x.device).manual_seed(50))  # noise in [0, 1]
        # noise = torch.rand(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print(ids_shuffle)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x.clone(), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        #-----------------------------------------------------------------------------------------------------------------
        if not typical:
            x_masked=x
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x_s=None,x_t=None ,mask_ratio_s=None,mask_ratio_t=None):
        # embed patches
        if x_s is not None:
            x_s = self.patch_embed(x_s)
            if x_t is not None:
                x_t = self.patch_embed(x_t)

            # add pos embed w/o cls token
        
            x_s = x_s + self.pos_embed[:, 1:, :]
            if x_t is not None:
                x_t = x_t + self.pos_embed_T[:, 1:, :]

            # masking: length -> length * mask_ratio
            
            
            if x_t is not None:
                x_s, mask_s, ids_restore_s = self.random_masking(x_s, mask_ratio_s)
                x_t, mask_t, ids_restore_t = self.random_masking(x_t, mask_ratio_t)
            else:
                x_s, mask_s, ids_restore_s = self.random_masking(x_s, mask_ratio_s,typical=True)
                mask_t=None
                ids_restore_t=None

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            
            cls_tokens = cls_token.expand(x_s.shape[0], -1, -1)
            x_s = torch.cat((cls_tokens, x_s), dim=1)
            if x_t is not None :
                cls_token = self.cls_token + self.pos_embed_T[:, :1, :]
                cls_tokens = cls_token.expand(x_s.shape[0], -1, -1)
                x_t = torch.cat((cls_tokens, x_t), dim=1)
                x=torch.cat((x_s,x_t),dim=1)
            else:
                x=x_s
        else:
            x_t = self.patch_embed(x_t)
            x_t = x_t + self.pos_embed_T_Seg[:, 1:, :]
            x_t, mask_t, ids_restore_t = self.random_masking(x_t, mask_ratio_t,typical=True)
            cls_token = self.cls_token + self.pos_embed_T_Seg[:, :1, :]
            cls_tokens = cls_token.expand(x_t.shape[0], -1, -1)
            x_t = torch.cat((cls_tokens, x_t), dim=1)
            x= x_t
            mask_s=None
            ids_restore_s=None
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)

        return x, mask_s,mask_t, ids_restore_s,ids_restore_t
    
    def forward_decoder_no_pred(self, x, ids_restore_s,ids_restore_t=None):
        # embed tokens
        # x = self.decoder_embed(x)
        if ids_restore_s is not None:
        
            x_s=x[:, :ids_restore_s.shape[1]+1, :]
            if ids_restore_t is not None:
                x_t=x[:, ids_restore_s.shape[1]+1:, :]

            # append mask tokens to sequence
            mask_tokens_s = self.mask_token.repeat(x_s.shape[0], ids_restore_s.shape[1] + 1 - x_s.shape[1], 1)
            x_s_ = torch.cat([x_s[:, 1:, :], mask_tokens_s], dim=1)  # no cls token
            x_s_ = torch.gather(x_s_, dim=1, index=ids_restore_s.unsqueeze(-1).repeat(1, 1, x_s.shape[2]))  # unshuffle
            x_s = torch.cat([x_s[:, :1, :], x_s_], dim=1)  # append cls token

            if ids_restore_t is not None:
                mask_tokens_t = self.mask_token.repeat(x_t.shape[0], ids_restore_t.shape[1] + 1 - x_t.shape[1], 1)
                x_t_ = torch.cat([x_t[:, 1:, :], mask_tokens_t], dim=1)
                x_t_ = torch.gather(x_t_, dim=1, index=ids_restore_t.unsqueeze(-1).repeat(1, 1, x_t.shape[2]))
                x_t = torch.cat([x_t[:, :1, :], x_t_], dim=1)

            # add pos embed
            x_s = x_s + self.decoder_pos_embed
            if ids_restore_t is not None:
                x_t = x_t + self.decoder_pos_embed_T

                x=torch.cat((x_s,x_t),dim=1)
            else:
                x=x_s
        else:
              x_t=x[:, :ids_restore_t.shape[1], :]
              mask_tokens_t = self.mask_token.repeat(x_t.shape[0], ids_restore_t.shape[1] + 1 - x_t.shape[1], 1)
              x_t_ = torch.cat([x_t[:, 1:, :], mask_tokens_t], dim=1)  # no cls token
              x_t_ = torch.gather(x_t_, dim=1, index=ids_restore_t.unsqueeze(-1).repeat(1, 1, x_t.shape[2]))  # unshuffle
              x_t = torch.cat([x_t[:, :1, :], x_t_], dim=1)  # append cls token
              x_t = x_t + self.decoder_pos_embed_T_Seg
              x=x_t
        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        # x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: B, C, T, H, W
        target: B, L, D
        pred: B, L, D
        mask: B, L. 0 is keep, 1 is remove,
        """
        # pred=pred[:,self.patch_embed.num_patches+1:]
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
