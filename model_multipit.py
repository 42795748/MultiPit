import math
from functools import partial
from typing import Tuple

import torch
from torch import nn

from utils_multipit import Block, to_2tuple, trunc_normal_


class SequentialTuple(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None, drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

        self.pool = pool

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, cls_tokens = x
        B, C, H, W = x.shape
        token_length = cls_tokens.shape[1]

        x = x.flatten(2).transpose(1, 2) # B,C,H,W -> B,H*W,C
        x = torch.cat((cls_tokens, x), dim=1) #B,1+H*W,C

        x = self.blocks(x)

        cls_tokens = x[:, :token_length] # B,1,C
        x = x[:, token_length:]
        x = x.transpose(1, 2).reshape(B, C, H, W)

        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)
        return x, cls_tokens


class ConvHeadPooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride,
            padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class Pit_Stages(nn.Module):
    def __init__(self, base_dims, depth, heads, mlp_ratio, attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0) -> None:
        super().__init__()
        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage])
            ]
        self.transformers = SequentialTuple(*transformers)
    
    def forward(self, x):
        x = self.transformers(x)
        return x

class MultiPit(nn.Module):
    def __init__(
            self, img_size, patch_size, stride, base_dims, depth, heads, branch_dims, branch_depth, branch_heads,
            mlp_ratio, num_parts=8, num_classes=1000, in_chans=3, attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(MultiPit, self).__init__()

        padding = 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        height = math.floor((img_size[0]*num_parts + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.num_tokens = 1
        self.num_parts = num_parts

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)

        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage])
            ]
        '''
        self.backbone = Pit_Stages(base_dims=base_dims, depth=depth, heads=heads,mlp_ratio=mlp_ratio,attn_drop_rate=attn_drop_rate,drop_rate=drop_rate, drop_path_rate= drop_path_rate)
        
        self.reverse_x = nn.Conv2d(base_dims[-1]*heads[-1],branch_dims[0]*branch_heads[0],3,1,1)
        self.reverse_cls = [ nn.Linear(base_dims[-1]*heads[-1],branch_dims[0]*branch_heads[0]) for _ in range(num_parts) ]
        self.reverse_cls = nn.ModuleList(self.reverse_cls)
        
        self.branch_nets = [Pit_Stages(base_dims=branch_dims, depth=branch_depth, heads=branch_heads,mlp_ratio=mlp_ratio,attn_drop_rate=attn_drop_rate,drop_rate=drop_rate, drop_path_rate= drop_path_rate) for _ in range(num_parts)]
        self.branch_nets = nn.ModuleList(self.branch_nets)
        
        self.norm = nn.LayerNorm(branch_dims[-1] * branch_heads[-1], eps=1e-6)
        self.num_features = self.embed_dim = branch_dims[-1] * branch_heads[-1]

        # Classifier head
        self.heads = [ nn.Linear(self.embed_dim, num_classes) for _ in range(num_parts) ]
        self.heads = nn.ModuleList(self.heads)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.heads = [ nn.Linear(self.embed_dim, num_classes) for _ in range(self.num_parts) ]
        self.heads = nn.ModuleList(self.heads)

    def forward_features(self, x:torch.Tensor):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x, cls_tokens = self.backbone((x, cls_tokens))
        
        x = self.reverse_x(x)
        cls_tokenss=[self.reverse_cls[i](cls_tokens) for i in range(self.num_parts)]
        
        _,_,H,_ = x.shape
        H = int(H/self.num_parts) # Take out the individual body part
        #TODO: 加上后续分支网络
        body_parts=[x[:,:,H*i:H*(i+1)] for i in range(self.num_parts)]
        # use cls_tokens as the common features extracted by backbone
        
        for i in range(self.num_parts):
            _, cls_tokenss[i] = self.branch_nets[i]((body_parts[i], cls_tokenss[i]))
            cls_tokenss[i] = self.norm(cls_tokenss[i]).squeeze(dim=1)
        return cls_tokenss

    def forward_heads(self, x, pre_logits: bool = False) -> torch.Tensor:
        if not pre_logits:
            for i in range(self.num_parts):
                x[i] = self.heads[i](x[i])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_heads(x)
        x = torch.stack(x,2) # (B,num_class,num_parts)
        return x