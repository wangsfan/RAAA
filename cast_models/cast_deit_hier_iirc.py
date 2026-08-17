"""Define CAST model for classification following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed
from timm.models.layers import trunc_normal_

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool_iirc import GraphPooling
from cast_models.modules_iirc import Pooling, ConvStem

class CAST(VisionTransformer):
    def __init__(self, *args, **kwargs):
        # These entries do not exist in timm.VisionTransformer.
        # 从kwargs中提取深度配置，这是一个包含每层深度的列表
        depths = kwargs['depth']
        # 从关键字参数中提取聚类数量，imagenet默认值为[64, 32, 16, 8]，并从kwargs中移除该参数
        num_clusters = kwargs.pop('num_clusters')
        # 将深度参数的值更新为其所有元素的总和，用于计算网络的总层数
        kwargs['depth'] = sum(kwargs['depth'])
        # 调用父类VisionTransformer的初始化方法
        super().__init__(**kwargs)

        # Do not tackle dist_token.
        # 验证蒸馏相关的组件未被使用，确保dist_token为None
        assert self.dist_token is None, 'dist_token is not None.'
        # 验证蒸馏头部未被使用，确保head_dist为None
        assert self.head_dist is None, 'head_dist is not None.'

        # 获取嵌入块的patch数量
        num_patches = self.patch_embed.num_patches
        # 初始化位置嵌入参数，形状为(1, num_patches, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        # 对位置嵌入参数进行截断正态分布初始化
        trunc_normal_(self.pos_embed, std=.02)

        # 计算累积深度，用于后续分割blocks
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        # 初始化blocks和pools列表，用于存储不同层级的注意力块和池化层
        blocks = []
        pools = []
        for ind, depth in enumerate(depths):
            # Build Attention Blocks.
            # 构建注意力块，根据累积深度切分总blocks
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind + 1]])

            # Build Pooling layers
            # 构建池化层
            pool = Pooling(
                pool_block=GraphPooling(
                    # 每个层级使用不同的聚类数量
                    num_clusters=num_clusters[ind],
                    # 使用嵌入维度作为图池化的模型维度
                    d_model=kwargs['embed_dim'],
                    # 不对FPS进行L2归一化
                    l2_normalize_for_fps=False))

            # 在dytox-hcast中或许需要梯度？因为我使用了每次池化后所得的centroid作为层级特征，并使用的是最后两层的centroid
            # # Last graph pooling is not needed.
            # # 最后一层的图池化不需要训练参数
            # if ind == len(depths) - 1:  # 为啥不直接没有pool4，而是把它的参数设为无梯度，它会用于前向传播吗？我认为连blocks4也不需要？总共最多三级，顶多只需要前三个blocks的输出特征即可？
            #     # 冻结最后一层池化层的fc1参数
            #     for param in pool.pool_block.fc1.parameters():
            #         param.requires_grad = False
            #     # 冻结最后一层池化层的fc2参数
            #     for param in pool.pool_block.fc2.parameters():
            #         param.requires_grad = False
            #     # 冻结最后一层池化层的质心全连接参数
            #     for param in pool.pool_block.centroid_fc.parameters():
            #         param.requires_grad = False
            pools.append(pool)

        # 将各层的blocks和pooling分配给实例属性
        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        # self.blocks4, self.pool4 = blocks[3], pools[3]
        # --------------------------------------------------------------------------

    def _block_operations(self, x, x_pad_mask,
                          nn_block, pool_block, norm_block):
        """Wrapper to define operations per block.
        """
        # Forward nn block with x
        # x(segment tokens)输入当前blocks
        x = nn_block(x).type_as(x)

        # Perform pooling only on x
        pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(x, x_pad_mask)
        )

        if norm_block is not None: #对最后一个Block的输出做norm_block(norm_layer:partial(nn.LayerNorm, eps=1e-6))
            centroid = norm_block(centroid)

        # 返回x: 当前blocks输出的segment tokens，[B, num_segments_{l-1}, embed_dim];
        # pool_logit: 尚未softmax的分配矩阵soft assignment probability matrix P_{l}, [B, num_segments_{l-1}, num_segments_l]；
        # centroid: 经过GraphPooling得到的聚类中心（即新segment tokens）, [B, num_segments_l, embed_dim]；
        # pool_pad_mask: 指示本次GraphPooling中填充的segment位置的mask（若是填充的则为True，实际上除了首次，其余均为False）, [B, num_segments_{l-1}]；
        # pool_inds: 聚类中心采样索引，指示了原segment tokens中哪些segment token被作为了聚类中心，[B, num_segments_{l-1}]；
        return (x, pool_logit, centroid, pool_pad_mask, pool_inds)

    def forward_features(self, x, y):  # x: B,C3,H,W   y: H,W
        x = self.patch_embed(x)
        # patch_embed层为cast_models.modules.ConvStem实例时x是4维的N H W C空间格式，而普通pe层得到的是展平的N H*W C格式
        # 故需要reshape
        H = W = int(x.shape[1] ** 0.5)  # 假设是方形特征图
        x = x.reshape(x.shape[0], H, W, -1)

        N, H, W, C = x.shape  # B, H224/patch_size8=28, H224/patch_size8=28, embed_dim384

        # Collect features within each segment
        # 使用最近邻插值实现将y的空间分辨率（即，(H,W)）变（下采样）为x.shape[1:3]（即，(H//8,W//8)），便于后期做均值池化
        y = y.unsqueeze(1).float()  # B,1,H,W
        y = F.interpolate(y, x.shape[1:3], mode='nearest')
        y = y.squeeze(1).long()
        # 取每个segment中所有像素特征的均值，作为该segment的初始视觉特征（ps：视觉特征而非特征，要后续叠加位置编码、拼接class token之后才得到真正的初始特征Z_{L+1}）
        x = segment_mean_nd(x, y)

        # Create padding mask
        ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5

        # Collect positional encodings within each segment
        pos_embed = self.pos_embed.view(1, H, W, C).expand(N, -1, -1, -1)
        pos_embed = segment_mean_nd(pos_embed, y)

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)

        # intermediate results
        intermediates = {}

        # Block1
        (block1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1) = self._block_operations(
            x, x_padding_mask,
            self.blocks1, self.pool1, None)

        intermediates1 = {
            'logit1': pool_logit1, 'centroid1': centroid1, 'block1': block1,
            'padding_mask1': x_padding_mask, 'sampled_inds1': pool_inds1,
        }
        intermediates.update(intermediates1)

        # Block2
        (block2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2) = self._block_operations(
            centroid1, pool_padding_mask1,
            self.blocks2, self.pool2, None)

        intermediates2 = {
            'logit2': pool_logit2, 'centroid2': centroid2, 'block2': block2,
                'padding_mask2': pool_padding_mask1, 'sampled_inds2': pool_inds2
        }
        intermediates.update(intermediates2)

        # Block3
        # (block3, pool_logit3, centroid3,
        #  pool_padding_mask3, pool_inds3) = self._block_operations(
        #     centroid2, pool_padding_mask2,
        #     self.blocks3, self.pool3, None)
        (block3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3) = self._block_operations(
            centroid2, pool_padding_mask2,
            self.blocks3, self.pool3, self.norm)

        intermediates3 = {
            'logit3': pool_logit3, 'centroid3': centroid3, 'block3': block3,
            'padding_mask3': pool_padding_mask2, 'sampled_inds3': pool_inds3
        }
        intermediates.update(intermediates3)

        # Block4
        # (block4, pool_logit4, centroid4,
        #  pool_padding_mask4, pool_inds4) = self._block_operations(
        #     centroid3, pool_padding_mask3,
        #     self.blocks4, self.pool4, self.norm)
        #
        # intermediates4 = {
        #     'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
        #     'padding_mask4': pool_padding_mask3, 'sampled_inds4': pool_inds4
        # }
        # intermediates.update(intermediates4)

        return intermediates

    def forward(self, x, y):
        intermediates = self.forward_features(x, y)
        # 2-level下，令centroid4为粗特征、centroid3为细，centroid2和centroid1不使用
        # return intermediates['centroid3'], intermediates['centroid4']
        return intermediates['centroid2'], intermediates['centroid3']



# def cast_small(pretrained=False, **kwargs):
#     # cifar
#     # minus one ViT block
#     model = CAST(
#         patch_size=2, embed_dim=384, num_clusters=[32, 16, 8, 4],
#         depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
#     # imagenet
#     model = CAST(
#         patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
#         depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
#     return model



