"""Define Graph Pooling."""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.geometry as dgl_geo
from timm.models.layers.drop import DropPath

from cast_models.utils import segment_mean_nd


class Attention(nn.Module):
    """Similar to timm.models.vision_transformer.Attention but we do not use
    additional Fully Connected Layers.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Same as timm.models.vision_transformer.Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(dim).normal_(0, 1e-2))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))

        x = x - torch.mean(x, dim=1, keepdim=True) + self.bias.view(1, 1, -1)

        return x


class GraphPooling(nn.Module):

    def __init__(self,
                 num_clusters=4, # [64, 32, 16, 8]
                 d_model=512, # embed_dim384
                 dropout=0.1,
                 l2_normalize_for_fps=True, # Fasle
                 num_heads=12,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        """Perfrom Graph Pooling.

        Args:
          num_clusters: A scalar indicates the number of centroids
          d_model: A scalar indicates the input channels to Transformer
          dropout: A `float` indicates the dropout rate
          l2_normalize_for_fps: Enable/disable L2-normalization before performing
                                Farthest Point Sampling
          num_heads: A scalar indicates the number of attention head
          qkv_bias: Enable/disable bias in the attention layer
          norm_layer: A Norm layer used in the attention layer
        """
        super().__init__()
        self.centroid_fc = Block(dim=d_model,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 norm_layer=norm_layer)
        self.fc1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=True),
            nn.GELU(),
            nn.Dropout(dropout))
        self.fc2 = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model, bias=True))

        self._num_clusters = num_clusters
        self._l2_normalize_for_fps = l2_normalize_for_fps

    def _fill_with_mean(self, src, mask):
        """A helper function to fill invalid entries with mean values.
        """
        bs, sl, cs = src.shape
        if mask is not None:
            mean_src = valid_mean(src, ~mask).unsqueeze(1).type_as(src)
            # Fill padded entries with mean values.
            fill_mask = mask.unsqueeze(2).expand(-1, -1, cs)
            filled_src = torch.where(fill_mask, mean_src.expand(-1, sl, -1), src)
        else:
            mean_src = torch.mean(src, dim=1, keepdim=True).type_as(src)
            filled_src = src

        return filled_src, mean_src

    def forward(self, src, mask):
        """Feedforward for clustering with Transformer.

        Args:
          src: A `tensor` of shape `[batch_size, source_sequence_length, channels]`
          mask: A bool `tensor` of shape `[batch_size, sequence_length]`, where
                `True` indicates empty/padded elements.

        Returns:
          cls_token: A `tensor` of shape `[batch_size, 1, channels]`
          centroids: A `tensor` of shape `[batch_size, num_clusters, channels]`
          logits: A `tensor` of shape
            `[batch_size, source_sequence_length, num_clusters]`
          sampled_inds: A `tensor` of shape
            `[batch_size, num_clusters]`
        """
        # 获取src的批次大小、序列长度（src_num_segments）和通道数(embed_dim)
        bs, sl, cs = src.shape

        # 使用最远点采样(FPS)从src中查询采样，选择出聚类中心，FPS能使采样点在空间中尽可能均匀分布
        # `centroids`的形状为`[batch_size, target_sequence_length(num_clusters), channels(embed_dim)]`
        # 使用_fill_with_mean函数填充无效项
        filled_src, mean_src = self._fill_with_mean(src, mask)
        # 将均值作为第一个token，拼接到填充后的src前面
        padded_src = torch.cat([mean_src, filled_src], dim=1)

        # 根据_l2_normalize_for_fps标志决定是否进行L2归一化
        if self._l2_normalize_for_fps:
            sampling_src = F.normalize(padded_src, dim=-1)
        else: # 执行该分支
            sampling_src = padded_src

        # 使用DGL几何库的最远点采样器选择聚类中心点
        sampled_inds = dgl_geo.farthest_point_sampler(
            sampling_src.to(torch.float64),  # 转换为float64精度
            self._num_clusters + 1,          # 采样数量（多采1个，后面会去掉）
            0).long()                       # 起始点索引
        # 去掉第一个采样点并调整索引（减去1），确保索引非负
        # 形状：[batch_size, num_clusters]，每个元素值指向原始 src 序列中一个具体位置，即采样点，进而构成聚类中心
        sampled_inds = sampled_inds[:, 1:] - 1
        # 断言确保所有索引都大于等于0
        assert((sampled_inds  >= 0).all())
        # 扩展采样索引维度以匹配通道数
        unfold_sampled_inds = sampled_inds.unsqueeze(2).expand(-1, -1, cs)

        # 应用注意力层预测分组
        # 通过centroid_fc Block处理src得到节点特征，可以视作Z_{l-1}前面的细特征（CAST和HCAST的顺序标号相反，此处采用CAST，即l-1相较l是更细层级）
        node_features = self.centroid_fc(src)
        # 根据采样索引收集节点特征作为聚类中心特征，可以视作Z_{l}后面的粗特征
        centroid_features = torch.gather(node_features, 1, unfold_sampled_inds)

        # 将序列token分组成聚类
        # 对聚类中心特征进行L2归一化
        normed_centroid_features = F.normalize(centroid_features, dim=-1)
        # 对节点特征进行L2归一化
        normed_node_features = F.normalize(node_features, dim=-1)
        # 计算节点特征与聚类中心特征之间的相似度
        logits = torch.einsum(
            'bij,bjk->bik', normed_node_features, normed_centroid_features.transpose(1, 2))
        # 将logits放大5倍以增强区分度，logits是尚未softmax的P_{l}
        logits = logits * 5
        # 使用softmax计算分配概率，即soft assignment probability matrix P_{l}
        assignments = torch.softmax(logits, dim=-1)

        # 在聚类内进行平均池化
        # 对src应用第一个全连接层，得到处理过的src特征
        fc1_src = self.fc1(src)

        # 批量矩阵乘法
        # 计算归一化因子
        normalizer = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2),
                                  torch.ones((bs, sl, 1), dtype=src.dtype, device=src.device))
        # 根据分配权重计算加权平均聚类中心
        centroids = torch.einsum('bij,bjk->bik', assignments.transpose(1, 2), fc1_src)
        # 归一化聚类中心
        centroids /= normalizer

        # 对聚类中心应用第二个全连接层，至此完成了CAST中的MLP(P^T*Z_{l-1}/P^T*1)
        fc2_centroids = self.fc2(centroids)
        # 更新聚类中心：添加残差连接（来自原始src的对应位置），至此完成了CAST中的Z_{l} = Z_{l-1} + MLP(P^T*Z_{l-1}/P^T*1)
        centroids = fc2_centroids + torch.gather(src, 1, unfold_sampled_inds)

        return centroids, logits, sampled_inds



def valid_mean(x, mask):
     """Compute mean of x given valid mask.

     Args:
         x: A `float` tensor of shape `[batch_size, num_nodes, channels]`
         mask: A `bool` tensor of shape `[batch_size, num_nodes]`, where
             `True` indicates the entry is valid/padded

     Returns:
         mean_x: A `float` tensor of shape `[batch_size, channels]`
     """
     mask = mask.type_as(x).unsqueeze(2)
     sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1)
     masked_x = x * mask
     mean_x = torch.sum(masked_x, dim=1) / sum_mask

     return mean_x

