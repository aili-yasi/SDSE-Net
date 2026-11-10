from einops import rearrange, reduce
from timm.layers.activations import *
from timm.layers import DropPath, trunc_normal_
from module.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D
import math
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import DropPath, create_act_layer, LayerType
import numpy as np
import torchvision
from typing import Callable, Dict, Optional, Type
from ccm.ccm import CCMix
import timm.layers.weight_init as weight_init

from functools import partial

T_MAX = 1024
inplace = True
wkv_cuda = load(
    name="wkv",
    sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={T_MAX}",
    ],
)



class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DWConv, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise Convolution
        x = self.pointwise(x)  # Pointwise Convolution
        return x


class DualConv3x3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConv3x3_block, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DWConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

# ====================== 从Mobile U-ViT移植的核心模块 ======================
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(
            in_features, hidden_features, 3, padding=1, groups=in_features
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, 3, padding=1, groups=in_features
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1.0,
    ):
        super().__init__()
        # ========== 修复1：强制保证通道数能被头数整除 ==========
        self.dim = dim
        self.base_num_heads = num_heads
        # 自动找到能整除dim的最大头数，避免reshape报错
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 如果不能整除，自动降级找合适的头数
        if dim % num_heads != 0:
            for possible_heads in range(min(num_heads, dim), 0, -1):
                if dim % possible_heads == 0:
                    self.num_heads = possible_heads
                    self.head_dim = dim // possible_heads
                    break
        # 缩放系数用修正后的head_dim计算
        self.scale = qk_scale or self.head_dim**-0.5

        # ========== 原有网络层 ==========
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio

        # 下采样+上采样层
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            self.LocalProp = nn.ConvTranspose2d(
                dim, dim, sr_ratio, stride=sr_ratio, groups=dim
            )
        else:
            self.sr = nn.Identity()
            self.norm = nn.Identity()
            self.LocalProp = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        # 保存原始尺寸，同时初始化下采样后的尺寸
        H_sr, W_sr = H, W
        x_sr = x  # 初始化下采样后的特征

        # ========== 修复2：下采样后严格记录新的尺寸，不再用原始N ==========
        if self.sr_ratio > 1:
            # 从序列还原回2D特征图做下采样
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x_sr_2d = self.sr(x_2d)
            # 记录下采样后的真实高宽
            H_sr, W_sr = x_sr_2d.shape[2], x_sr_2d.shape[3]
            # 转回序列格式
            x_sr = x_sr_2d.flatten(2).transpose(1, 2)
            x_sr = self.norm(x_sr)
        else:
            # 无下采样时，直接用原始输入
            x_sr = x

        # ========== 修复3：用修正后的num_heads和head_dim做reshape ==========
        qkv = (
            self.qkv(x_sr)
            .reshape(B, -1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ========== 修复4：用下采样后的真实序列长度reshape ==========
        if self.sr_ratio > 1:
            N_sr = H_sr * W_sr  # 下采样后的真实token数
            x = (attn @ v).transpose(1, 2).reshape(B, N_sr, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 上采样还原回原始尺寸
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H_sr, W_sr)
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (
            self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))))
            - 0.5
        )
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x


class SelfAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1.0,
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        # 位置编码（格式适配，不改变核心）
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        # 格式转换：2D特征图 → 1D序列（为了适配注意力的输入要求）
        x = x.flatten(2).transpose(1, 2)    # (B, C, H, W) → (B, H*W, C)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 格式转换：1D序列 → 2D特征图（为了适配后面的解码器）
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class LGLBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1.0,
    ):
        super().__init__()
        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer)
        else:
            self.LocalAgg = nn.Identity()
        self.SelfAttn = SelfAttn(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            sr_ratio,
        )

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SE(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        act_layer: LayerType = nn.ReLU,
        gate_layer: LayerType = nn.Sigmoid,
        force_act_layer: Optional[LayerType] = None,
        rd_round_fn: Optional[Callable] = None,
    ):
        super(SE, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        wkv_cuda.backward(
            B,
            T,
            C,
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
            gw,
            gu,
            gk,
            gv,
        )
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (
                None,
                None,
                None,
                gw.bfloat16(),
                gu.bfloat16(),
                gk.bfloat16(),
                gv.bfloat16(),
            )
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(
        B, C, patch_resolution[0], patch_resolution[1]
    )
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0 : int(C * gamma), :, shift_pixel:W] = input[
        :, 0 : int(C * gamma), :, 0 : W - shift_pixel
    ]
    output[:, int(C * gamma) : int(C * gamma * 2), :, 0 : W - shift_pixel] = input[
        :, int(C * gamma) : int(C * gamma * 2), :, shift_pixel:W
    ]
    output[:, int(C * gamma * 2) : int(C * gamma * 3), shift_pixel:H, :] = input[
        :, int(C * gamma * 2) : int(C * gamma * 3), 0 : H - shift_pixel, :
    ]
    output[:, int(C * gamma * 3) : int(C * gamma * 4), 0 : H - shift_pixel, :] = input[
        :, int(C * gamma * 3) : int(C * gamma * 4), shift_pixel:H, :
    ]
    output[:, int(C * gamma * 4) :, ...] = input[:, int(C * gamma * 4) :, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        # Use xk, xv, xr to produce k, v, r
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device
        sr, k, v = self.jit_func(x, patch_resolution)
        x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class GLSP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=True,
        has_skip=True,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        act_layer="relu",
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        attn_s=True,
        drop_path=0.0,
        drop=0.0,
        img_size=224,
        channel_gamma=1 / 4,
        shift_pixel=1,
    ):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.ln2 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel)
        self.se = (
            SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer))
            if se_ratio > 0.0
            else nn.Identity()
        )
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(
            dim_mid,
            dim_out,
            kernel_size=1,
            norm_layer="none",
            act_layer="none",
            inplace=inplace,
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.attn_s = attn_s
        # Set position embedding
        self.conv_local = ConvNormAct(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            dilation=dilation,
            groups=dim_mid,
            norm_layer="bn_2d",
            act_layer="silu",
            inplace=inplace,
        )
        self.drop_after_pos = nn.Dropout(p=0.1)
        self.num_extra_tokens = 0

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        if self.attn_s:
            B, hidden, H, W = x.size()
            patch_resolution = (H, W)
            x = x.view(B, hidden, -1)  # (B, hidden, H*W) = (B, C, N)
            x = x.permute(0, 2, 1)
            x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
            B, n_patch, hidden = (
                x.size()
            )  # reshape from (B, n_patch, hidden) to (B, h, w, hidde
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            x = x.permute(0, 2, 1)
            x = x.contiguous().view(B, hidden, h, w)
        x = (
            x + self.se(self.conv_local(x))
            if self.has_skip
            else self.se(self.conv_local(x))
        )
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class RWKV_UNet_encoder(nn.Module):
    def __init__(
        self,
        dim_in=3,
        num_classes=1000,
        img_size=224,
        depths=[2, 4, 4, 2],
        stem_dim=16,
        embed_dims=[64, 128, 256, 512],
        exp_ratios=[2.0, 2.0, 4.0, 4.0],
        norm_layers=["bn_2d", "bn_2d", "bn_2d", "bn_2d"],
        act_layers=["relu", "relu", "relu", "relu"],
        dw_kss=[3, 3, 1, 1],
        se_ratios=[0.0, 0.0, 0.0, 0.0],
        attn_ss=[False, False, True, True],
        drop=0.0,
        drop_path=0.0,
        pre_dim=0,
        channel_gamma=1 / 4,
        shift_pixel=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.embed_dims = embed_dims
        self.stage0 = nn.ModuleList(
            [
                GLSP(  # ds
                    dim_in,
                    stem_dim,
                    norm_in=False,
                    has_skip=False,
                    exp_ratio=1,
                    norm_layer=norm_layers[0],
                    act_layer=act_layers[0],
                    dw_ks=dw_kss[0],
                    stride=1,
                    dilation=1,
                    se_ratio=1,
                    attn_s=False,
                    drop_path=0.0,
                    drop=0.0,
                    img_size=224,
                )
            ]
        )
        img_size = img_size // 2
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]) : sum(depths[: i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = (
                        2,
                        False,
                        False,
                        exp_ratios[i] * 2,
                    )
                    img_size = img_size // 2
                else:
                    stride, has_skip, attn_s, exp_ratio = (
                        1,
                        True,
                        attn_ss[i],
                        exp_ratios[i],
                    )
                layers.append(
                    GLSP(
                        emb_dim_pre,
                        embed_dims[i],
                        norm_in=True,
                        has_skip=has_skip,
                        exp_ratio=exp_ratio,
                        norm_layer=norm_layers[i],
                        act_layer=act_layers[i],
                        dw_ks=dw_kss[i],
                        stride=stride,
                        dilation=1,
                        se_ratio=se_ratios[i],
                        attn_s=attn_s,
                        drop_path=dpr[j],
                        drop=drop,
                        img_size=img_size,
                    )
                )
                emb_dim_pre = embed_dims[i]
                if j == 0 & i > 1:
                    patch_size = patch_size // 2
            self.__setattr__(f"stage{i + 1}", nn.ModuleList(layers))

        # ====================== 【核心修改】正确适配Mobile U-ViT结构 ======================
        self.stage4_use_mobile = True
        if self.stage4_use_mobile:
            # ========== 新增：自动计算正确头数的辅助函数，杜绝整除问题 ==========
            def _get_valid_num_heads(dim, max_head_count=16):
                # 优先用64作为单头通道数，不行就自动找能整除的最大头数
                if dim % 64 == 0:
                    return dim // 64
                # 兜底：找不超过max_head_count、能整除dim的最大头数
                for head_num in range(min(max_head_count, dim), 0, -1):
                    if dim % head_num == 0:
                        return head_num
                return 1  # 极端情况保底

            # 1. GLSP块×3 (在 h/8, w/8 上，sr_ratio=2)
            #    对应Mobile U-ViT的 adaptive_lgl_bottleneck
            self.glsp_blocks = nn.ModuleList(
                [
                    LGLBlock(
                        dim=embed_dims[2],  # Stage 3的输出维度，h/8,w/8
                        num_heads=_get_valid_num_heads(
                            embed_dims[2]
                        ),  # 自动计算正确头数
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=drop,
                        attn_drop=0.0,
                        drop_path=(
                            dprs[sum(depths[:2]) + i]
                            if sum(depths[:2]) + i < len(dprs)
                            else 0.1
                        ),
                        act_layer=get_act(act_layers[2]),
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        sr_ratio=2,  # GLSP固定用sr_ratio=2
                    )
                    for i in range(depths[2])
                ]
            )

            # # 2. LKLGL块×3 (在 h/16, w/16 上，sr_ratio=1)
            # self.lklgl_blocks = nn.ModuleList(
            #     [
            #         LGLBlock(
            #             dim=embed_dims[3],  # Stage4的输出维度，h/16,w/16
            #             num_heads=_get_valid_num_heads(
            #                 embed_dims[3]
            #             ),  # 自动计算正确头数
            #             mlp_ratio=4.0,
            #             qkv_bias=True,
            #             qk_scale=None,
            #             drop=drop,
            #             attn_drop=0.0,
            #             drop_path=0.1,
            #             act_layer=get_act(act_layers[3]),
            #             norm_layer=partial(nn.LayerNorm, eps=1e-6),
            #             sr_ratio=1,  # 高分辨率低的地方不用下采样
            #         )
            #         for i in range(depths[3])
            #     ]
            # )

            # 3. 通道升维 expend_dims
            self.expend_dims = ConvBlock(ch_in=embed_dims[3], ch_out=embed_dims[3] * 2)

            # 4. Transformer Bottleneck×3 (sr_ratio=1)
            self.transformer_bottleneck = nn.ModuleList(
                [
                    LGLBlock(
                        dim=embed_dims[3] * 2,  # 升维后的通道数
                        num_heads=_get_valid_num_heads(
                            embed_dims[3] * 2
                        ),  # 自动计算正确头数
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=drop,
                        attn_drop=0.0,
                        drop_path=0.1,
                        act_layer=get_act(act_layers[3]),
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        sr_ratio=1,
                    )
                    for _ in range(1)
                ]
            )

            # 5. 通道降维 reduce_dims
            self.reduce_dims = ConvBlock(ch_in=embed_dims[3] * 2, ch_out=embed_dims[3])
        # ====================== 修改结束 ======================
        self.pre_dim = embed_dims[-1]
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m,
            (
                nn.LayerNorm,
                nn.GroupNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"token"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"alpha", "gamma", "beta"}

    @torch.jit.ignore
    def no_ft_keywords(self):
        # return {'head.weight', 'head.bias'}
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self):
        return {"head.weight", "head.bias"}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(
                    m.running_mean, nan=0, posinf=1, neginf=-1
                )
                m.running_var = torch.nan_to_num(
                    m.running_var, nan=0, posinf=1, neginf=-1
                )

    # 辅助函数：找到合适的 num_heads
    def _get_num_heads(dim, base=64):
        if dim % base == 0:
            return dim // base
        # 寻找能整除 dim 的最大可能值，接近 dim//base
        target = dim // base
        for possible in range(target, 0, -1):
            if dim % possible == 0:
                return possible
        return 1  # 保底

    def forward_features(self, x):
        for blk in self.stage0:
            x = blk(x)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        for blk in self.stage4:
            x = blk(x)
        # # ====================== 【核心修改】Stage IV 前向逻辑 ======================
        # if hasattr(self, "stage4_use_mobile") and self.stage4_use_mobile:
        #     # 1. 先通过原始的stage4进行下采样和初步处理
        #     for blk in self.stage4:
        #         x = blk(x)

        #     # 2. LKLGL块×3 (在 h/16, w/16 尺度上)
        #     for blk in self.lklgl_blocks:
        #         x = blk(x)

        #     # 3. 通道升维
        #     x = self.expend_dims(x)

        #     # 4. Transformer Bottleneck×3
        #     for blk in self.transformer_bottleneck:
        #         x = blk(x)

        #     # 5. 通道降维
        #     x = self.reduce_dims(x)
        # # ====================== 修改结束 ======================
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def RWKV_UNet_encoder_T(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[2, 2, 4, 2],
        stem_dim=24,
        embed_dims=[32, 48, 96, 160],
        exp_ratios=[2.0, 2.5, 3.0, 3.5],
        norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
        act_layers=["silu", "silu", "gelu", "gelu"],
        dw_kss=[5, 5, 5, 5],
        attn_ss=[False, False, True, True],
        drop=0.0,
        drop_path=0.05,
        **kwargs,
    )
    return model


def RWKV_UNet_encoder_S(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 6, 3],
        stem_dim=24,
        embed_dims=[32, 64, 128, 192],
        exp_ratios=[2.0, 2.5, 3.0, 4.0],
        norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
        act_layers=["silu", "silu", "gelu", "gelu"],
        dw_kss=[5, 5, 5, 5],
        attn_ss=[False, False, True, True],
        drop=0.0,
        drop_path=0.05,
        **kwargs,
    )
    return model


def RWKV_UNet_encoder_B(pretrained=False, **kwargs):
    model = RWKV_UNet_encoder(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 6, 3],
        stem_dim=24,
        embed_dims=[48, 72, 144, 240],
        exp_ratios=[2.0, 2.5, 4.0, 4.0],
        
        # U-Bench中全是bn_2d
        # norm_layers=["bn_2d", "bn_2d", "bn_2d", "bn_2d"],
        norm_layers=["bn_2d", "bn_2d", "ln_2d", "ln_2d"],
        act_layers=["silu", "silu", "gelu", "gelu"],
        dw_kss=[5, 5, 5, 5],
        attn_ss=[False, False, True, True],
        drop=0.0,
        drop_path=0.05,
        **kwargs,
    )
    return model


class UpBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=False,
        has_skip=False,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        drop_path=0.0,
        drop=0.0,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)

        # ========== 核心修改：替换原单分支结构为DPGS-Net EFD模块（DualConv3x3_block） ==========
        # 原代码单分支卷积结构，替换为带残差的EFD模块
        self.efd_block = DualConv3x3_block(dim_in, dim_mid)
        # ==========================================================================================

        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.se = (
            SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer))
            if se_ratio > 0.0
            else nn.Identity()
        )
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(
            dim_mid,
            dim_out,
            kernel_size=1,
            norm_layer="bn_2d",
            act_layer="relu",
            inplace=inplace,
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        # self.conv_local = ConvNormAct(
        #     dim_mid,
        #     dim_mid,
        #     kernel_size=dw_ks,
        #     stride=stride,
        #     dilation=dilation,
        #     groups=dim_mid,
        #     norm_layer="bn_2d",
        #     act_layer="silu",
        #     inplace=inplace,
        # )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.efd_block(x)
        x = (
            x + self.se(x)
            if self.has_skip
            else self.se(x)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x


class SDSE_Net(nn.Module):
    def __init__(
        self, in_channels=1, num_classes=9, img_size=224, pretrained_path="net_B.pth"
    ):
        super(SDSE_Net, self).__init__()
        self.encoder = RWKV_UNet_encoder_B(img_size=img_size)
        if pretrained_path:
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(pretrained_dict, strict=False)

        self.embed_dims = [48, 72, 144, 240]
        self.ccm = CCMix(
            [self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
            self.embed_dims[0],
            img_size // 2,
        )
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(
            self.embed_dims[3],
            self.embed_dims[2],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder2 = UpBlock(
            self.embed_dims[2] * 2,
            self.embed_dims[1],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder3 = UpBlock(
            self.embed_dims[1] * 2,
            self.embed_dims[0],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder4 = UpBlock(
            self.embed_dims[0] * 2,
            24,
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        input = x
        x = x.repeat(1, 3, 1, 1)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        # ====================== 【核心修改】Stage IV 前向逻辑 ======================
        # 1. 先通过原始的stage4进行下采样
        for blk in self.encoder.stage4:
            x = blk(x)

        # # 2. LKLGL块×3
        # for blk in self.encoder.lklgl_blocks:
        #     x = blk(x)

        # 3. 通道升维
        x = self.encoder.expend_dims(x)

        # 4. Transformer Bottleneck×3
        for blk in self.encoder.transformer_bottleneck:
            x = blk(x)

        # 5. 通道降维
        x = self.encoder.reduce_dims(x)
        # ====================== 修改结束 ======================
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out


class RWKV_UNet_S(nn.Module):
    def __init__(
        self, in_channels=1, num_classes=9, img_size=224, pretrained_path="net_S.pth"
    ):
        super(RWKV_UNet_S, self).__init__()
        self.encoder = RWKV_UNet_encoder_S(img_size=img_size)
        if pretrained_path:
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(pretrained_dict, strict=False)
        self.embed_dims = [32, 64, 128, 192]
        self.ccm = CCMix(
            [self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
            self.embed_dims[0],
            img_size // 2,
        )
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(
            self.embed_dims[3],
            self.embed_dims[2],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder2 = UpBlock(
            self.embed_dims[2] * 2,
            self.embed_dims[1],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder3 = UpBlock(
            self.embed_dims[1] * 2,
            self.embed_dims[0],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder4 = UpBlock(
            self.embed_dims[0] * 2,
            24,
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        input = x
        x = x.repeat(1, 3, 1, 1)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out


class RWKV_UNet_T(nn.Module):
    def __init__(
        self, in_channels=1, num_classes=1, img_size=256, pretrained_path="net_T.pth"
    ):
        super(RWKV_UNet_T, self).__init__()
        self.encoder = RWKV_UNet_encoder_T(img_size=img_size)
        if pretrained_path:
            model_dict = self.encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")
            self.encoder.load_state_dict(pretrained_dict, strict=False)

        self.embed_dims = [32, 48, 96, 160]
        self.ccm = CCMix(
            [self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
            self.embed_dims[0],
            img_size // 2,
        )
        # Decoder layers using iRWKV and 1x1 Conv with bilinear upsampling
        self.decoder1 = UpBlock(
            self.embed_dims[3],
            self.embed_dims[2],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder2 = UpBlock(
            self.embed_dims[2] * 2,
            self.embed_dims[1],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder3 = UpBlock(
            self.embed_dims[1] * 2,
            self.embed_dims[0],
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.decoder4 = UpBlock(
            self.embed_dims[0] * 2,
            24,
            norm_in=False,
            has_skip=False,
            exp_ratio=1.0,
            dw_ks=9,
            stride=1,
            dilation=1,
            se_ratio=0.0,
            drop_path=0.0,
            drop=0.0,
        )
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        input = x
        # x = x.repeat(1, 3, 1, 1)
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x
        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x
        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x
        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x
        for blk in self.encoder.stage4:
            x = blk(x)
        enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])
        # Decoder path with concatenated skip connections
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return out
