# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

from models.swin_transformer_v2 import *
import torch
import torch.nn as nn
from einops import rearrange


class Mlp_head(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.norm = nn.LayerNorm(out_features)
        self.act2 = nn.Hardtanh(0, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # x = self.norm(x)
        x = self.act2(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'B h w (p1 p2 c)-> B (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'B h w (p1 p2 c)-> B (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x1, x2):
        # x1 and x2 are tensors of shape (N, HW, C) and (N, HW, C) respectively
        # N is the batch size, C is the number of channels, H is the height and W is the width
        # The output tensor will have shape (N,H*W,C1 + C2)
        return torch.cat((x1, x2), dim=-1)


class SwinMFF(nn.Module):
    r""" SwinMFIF
           A PyTorch impl of :
       Args:
           img_size (int | tuple(int)): Input image size. Default 224
           patch_size (int | tuple(int)): Patch size. Default: 4
           in_chans (int): Number of input image channels. Default: 3
           num_classes (int): Number of classes for classification head. Default: 1000
           embed_dim (int): Patch embedding dimension. Default: 96
           depths (tuple(int)): Depth of each Swin Transformer layer.
           num_heads (tuple(int)): Number of attention heads in different layers.
           window_size (int): Window size. Default: 7
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
           qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
           drop_rate (float): Dropout rate. Default: 0
           attn_drop_rate (float): Attention dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
       """

    def __init__(self, img_size=256, patch_size=4, in_chans=1, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 6, 2, 2], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  # drop_path_rate=0.1
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[8, 8, 8, 8,8,8], **kwargs):
        super(SwinMFF, self).__init__()

        self.dim_input_list = [embed_dim, embed_dim * 2, embed_dim * 4,embed_dim * 8,
                               4*embed_dim,2*embed_dim]  # [encode encode fusion decode decode]
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.input_resolution_list = [self.patches_resolution,
                                      (self.patches_resolution[0] // 2, self.patches_resolution[1] // 2),
                                      (self.patches_resolution[0] // 4, self.patches_resolution[1] // 4),
                                      (self.patches_resolution[0] // 4, self.patches_resolution[1] // 4),
                                      (self.patches_resolution[0] // 2, self.patches_resolution[1] // 2),
                                      self.patches_resolution]
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_classes = num_classes
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build swin transformer layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=self.dim_input_list[i_layer],
                               input_resolution=self.input_resolution_list[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer == 0 or i_layer == 1)  else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        # build concat layer
        self.concat_layer = ConcatLayer()

        self.expand1 = PatchExpand(input_resolution=(patches_resolution[0] // 4, patches_resolution[1] // 4),dim=8*embed_dim)
        self.expand2 = PatchExpand(input_resolution=(patches_resolution[0] // 2, patches_resolution[1] // 2),dim=4*embed_dim)
        self.expand3 = FinalPatchExpand_X4(input_resolution=(patches_resolution[0], patches_resolution[1]),
                                           dim=embed_dim*2, dim_scale=4,
                                           norm_layer=norm_layer)

        # self.head=nn.Linear(embed_dim,self.num_classes)#rgb
        self.head = Mlp_head(2*embed_dim, embed_dim, 1)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2):
        ori_x1 = x1

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        for layer in self.layers[0:3]:
            x1 = layer(x1)
            x2 = layer(x2)
        x_concat = self.concat_layer(x1, x2)

        x_concat = self.layers[3](x_concat)
        x_concat_expand1 = self.expand1(x_concat)
        x_decode1 = self.layers[4](x_concat_expand1)
        x_concat_expand2 = self.expand2(x_decode1)
        x_decode2 = self.layers[5](x_concat_expand2)
        x_concat_expand3 = self.expand3(x_decode2)
        out = self.head(x_concat_expand3)

        B, C, H, W = ori_x1.shape
        out = out.permute(0, 2, 1).view(B, 1, H, W)  # [B,1,H,W]
        return out
# #
# input1 = torch.rand(1, 1, 256, 256)
# input2 = torch.rand(1, 1, 256, 256)
# model=SwinMFIF_Net()
# # print('Number of models parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
# out=model(input1,input2)
# #
# print(out.shape)
