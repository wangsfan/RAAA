import copy
import json
import os

import torch
from timm.models.layers import trunc_normal_
from torch import nn

from continual.cnn import resnet18
import continual.utils as cutils
from continual.convit import ClassAttention, Block
from continual.vit import VisionTransformer



class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def forward_slice(self, x, mask):
        x = self.norm(x)
        sliced_weight = self.head.weight[mask, :]  # 这里创建了原weight的视图(view)
        sliced_bias = self.head.bias[mask] if self.head.bias is not None else None

        output = torch.matmul(x, sliced_weight.t())  # 通过视图计算
        if sliced_bias is not None:
            output = output + sliced_bias
        return output

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n


class DyTox(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    # def __init__(
    #     self,
    #     transformer,
    #     nb_classes,
    #     individual_classifier='',
    #     head_div=False,
    #     head_div_mode=['tr', 'ft'],
    #     joint_tokens=False,
    #     resnet=False, dataset_name='iirc_cifar100', class_names_to_idx=None
    # ):
    def __init__(self, transformer, args):
        super().__init__()

        self.dataset_name = args.data_set
        self.class_names_to_idx = args.class_names_to_idx
        _, self.sub_idx_to_sup_idx = get_true_heirarchy_dict(self.dataset_name, self.class_names_to_idx)

        # 初始化全部任务上的子类mask:all_subclasses_mask，是一个长度为self.nb_total_classes的bool列表
        # 根据self.sub_idx_to_sup_idx中所有keys，即每个subclass对应的索引sub_idx使得all_subclasses_mask[sub_idx]为True
        # 并取反得到all_superclasses_mask
        self.nb_total_classes = args.nb_classes
        self.all_subclasses_mask = torch.zeros(self.nb_total_classes, dtype=torch.bool)
        for sub_idx in self.sub_idx_to_sup_idx.keys():
            self.all_subclasses_mask[sub_idx] = True
        # self.all_superclasses_mask = ~self.all_subclasses_mask

        self.nb_classes = args.initial_increment
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = args.ind_clf
        self.use_head_div = args.head_div > 0
        self.head_div_mode = args.head_div_mode
        self.head_div = None
        self.joint_tokens = args.joint_tokens
        self.in_finetuning = False

        self.use_resnet = args.resnet

        self.nb_classes_per_task = [self.nb_classes] # 记录了截至当前任务的各任务上类别数量，随任务进行动态扩展，不同于self.increment_per_task

        # 到每个新任务时要动态更新（扩展）
        self.up2cur_subclasses_mask = self.all_subclasses_mask[:sum(self.nb_classes_per_task)]

        self.increment_per_task = args.increment_per_task
        # self.sub_mask_per_task对应了self.all_subclasses_mask划分到各个任务上的mask构成的列表，len同self.increment_per_task，即nb_tasks
        self.sub_mask_per_task = []
        # self.super_mask_per_task = []
        start_idx = 0
        for increment in self.increment_per_task:
            end_idx = start_idx + increment
            self.sub_mask_per_task.append(self.all_subclasses_mask[start_idx:end_idx])
            # self.super_mask_per_task.append(self.all_superclasses_mask[start_idx:end_idx])
            start_idx = end_idx

        if self.use_resnet:
            print('ResNet18 backbone for ens')
            self.backbone = resnet18()
            self.backbone.head = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 504, kernel_size=1),
                nn.BatchNorm2d(504),
                nn.ReLU(inplace=True)
            )
            self.backbone.avgpool = nn.Identity()
            self.backbone.layer4 = nn.Identity()
            #self.backbone.layer4 = self.backbone._make_layer_nodown(
            #    256, 512, 2, stride=1, dilation=2
            #)
            self.backbone = self.backbone.cuda()
            self.backbone.embed_dim = 504
            self.embed_dim = self.backbone.embed_dim

            self.tabs = nn.ModuleList([
                Block(
                    dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                    attention_type=ClassAttention
                ).cuda()
            ])
            self.tabs[0].reset_parameters()

            token = nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda())
            trunc_normal_(token, std=.02)
            self.task_tokens = nn.ParameterList([token])
        else:
            # self.patch_embed = transformer.patch_embed
            # self.pos_embed = transformer.pos_embed
            # self.pos_drop = transformer.pos_drop
            # #
            # # if isinstance(transformer, VisionTransformer):
            # #     self.sabs = transformer.blocks[:5]
            # #     self.tabs = transformer.blocks[5:]
            # # else:
            # #     self.sabs = transformer.blocks[:transformer.local_up_to_layer]
            # #     self.tabs = transformer.blocks[transformer.local_up_to_layer:]
            # self.sabs = transformer.blocks[:transformer.local_up_to_layer]
            # self.tabs = transformer.blocks[transformer.local_up_to_layer:]
            #
            # self.task_tokens = nn.ParameterList([transformer.cls_token])

            self.backbone = transformer

            self.tabs = nn.ModuleList([
                Block(
                    dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                    attention_type=ClassAttention
                ).cuda()
            ])
            self.tabs[0].reset_parameters()

            token = nn.Parameter(torch.zeros(1, 1, self.embed_dim).cuda())
            trunc_normal_(token, std=.02)
            self.task_tokens = nn.ParameterList([token])

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    def end_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        self.up2cur_subclasses_mask = self.all_subclasses_mask[:sum(self.nb_classes_per_task)]

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            self.head_div = ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        if self.individual_classifier == '1-1': #dytox选择该分支
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'encoder':
                    self.backbone.eval()
                    cutils.freeze_parameters(self.backbone, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            # 错了吧，应该是取self.head[:-1]? freeze里也是-1
            # 'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
            #                   if self.individual_classifier else \
            #                   self.head.parameters(),
            'old_heads': self.head[:-1].parameters() \
                if self.individual_classifier else \
                self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None
        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features(self, x, y):#此处y指的是segments而非labels
        # Shared part, this is the ENCODER
        # x shape: (B, C, H, W)
        B = x.shape[0]

        if self.use_resnet:
            # x shape after forward_tokens:  feats contains intermediate features
            x, self.feats = self.backbone.forward_tokens(x)
        else:
            # 将图像和超像素输入backbone(cast模型)即encoder，得到每个样本的粗、细特征
            x_fine, x_coarse = self.backbone(x,y) #形状不同，粗特征为(B, num_segments_l, D)，细特征为(B, num_segments_{l-1}, D)，且num_segments_{l-1} > num_segments_l

        # 形状为B的一维bool张量用于指示样本是父类还是子类样本，
        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens: # 不执行
            return self.forward_features_jointtokens(x) # 将所有tasks上的的交叉注意力同时而非用循环依次计算，按理说与当前forward_features方法所得相同，实际性能差，作者也不知原因

        fine_tokens, coarse_tokens = [], []

        attentions = []
        mask_heads = None

        for task_token in self.task_tokens: # 依次使用各任务特定的task token经tab做交叉注意力
            # 广播，Expand task token to match batch size
            # task_token shape: (1, 1, D) -> (B, 1, D)
            fine_task_token = task_token.expand(B, -1, -1)
            coarse_task_token = fine_task_token

            for blk in self.tabs:
                # Concatenate task token with image patches
                # task_token shape: (B, 1+N, D)?  attn shape: (B, num_heads, 1+N, 1+N)
                fine_task_token, _, _ = blk(torch.cat((fine_task_token, x_fine), dim=1), mask_heads=mask_heads)
                coarse_task_token, _, _ = blk(torch.cat((coarse_task_token, x_coarse), dim=1), mask_heads=mask_heads)

            fine_tokens.append(fine_task_token[:, 0]) # tokens的长度为任务数量
            coarse_tokens.append(coarse_task_token[:, 0]) # tokens的长度为任务数量

        # fine_tokens/coarse_tokens: !list! of (B, D) tensors, one per task, length is n_tasks
        # tokens[-1]: (B, D), last task token
        return fine_tokens, fine_tokens[-1], coarse_tokens, coarse_tokens[-1]

    # def forward_features_jointtokens(self, x):
    #     """Method to do a single TAB forward with all task tokens.
    #
    #     A masking is used to avoid interaction between tasks. In theory it should
    #     give the same results as multiple TAB forward, but in practice it's a little
    #     bit worse, not sure why. So if you have an idea, please tell me!
    #     """
    #     B = len(x)
    #
    #     task_tokens = torch.cat(
    #         [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
    #         dim=1
    #     )
    #
    #     for blk in self.tabs:
    #         task_tokens, _, _ = blk(
    #             torch.cat((task_tokens, x), dim=1),
    #             task_index=len(self.task_tokens),
    #             attn_mask=True
    #         )
    #
    #     if self.individual_classifier in ('1-1', '1-n'):
    #         return task_tokens.permute(1, 0, 2), task_tokens[:, -1], None
    #     return task_tokens.view(B, -1), task_tokens[:, -1], None

    def forward_classifier(self, fine_tokens, last_fine_token, coarse_tokens, last_coarse_token):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param tokens: A list of all task tokens embeddings.
        :param last_token: The ultimate task token embedding from the latest task.
        """
        logits_div = None

        if self.individual_classifier != '':
            fine_logits, coarse_logits = [], []
            for i, head in enumerate(self.head):
                if self.individual_classifier in ('1-n', '1-1'):
                    fine_logits.append(head.forward_slice(fine_tokens[i], self.sub_mask_per_task[i]))
                    coarse_logits.append(head.forward_slice(coarse_tokens[i], ~self.sub_mask_per_task[i]))
                else:  # n-1, n-n
                    fine_logits.append(head.forward_slice(torch.cat(fine_tokens[:i+1], dim=1), self.sub_mask_per_task[i]))
                    coarse_logits.append(head.forward_slice(torch.cat(coarse_tokens[:i+1], dim=1), ~self.sub_mask_per_task[i]))

            if self.individual_classifier in ('1-1', 'n-1'):
                fine_logits = torch.cat(fine_logits, dim=1)
                coarse_logits = torch.cat(coarse_logits, dim=1)

            # dytox原代码有误？不理解，故此处不知如何修改，反正用不着
            # else:  # 1-n, n-n
            #     final_fine_logits = torch.zeros_like(fine_logits[-1])
            #     final_coarse_logits = torch.zeros_like(coarse_logits[-1])
            #     for i in range(len(fine_logits)):
            #         final_fine_logits[:, :fine_logits[i].shape[1]] += fine_logits[i]
            #         final_coarse_logits[:, :coarse_logits[i].shape[1]] += coarse_logits[i]
            #
            #     for i, c in enumerate(self.nb_classes_per_task):
            #         final_fine_logits[:, :c] /= len(self.nb_classes_per_task) - i
            #         final_coarse_logits[:, :c] /= len(self.nb_classes_per_task) - i
            #
            #     logits = final_fine_logits + final_coarse_logits
        # tokens不是list而是单张量？
        # elif isinstance(tokens, torch.Tensor):
        #     logits = self.head(tokens)
        # else:
        #     logits = self.head(torch.cat(tokens, dim=1))

        # 鉴于效果较差故dytox-hcast暂时不使用head_div
        # CIFAR中head_div_mode为'tr'，故微调时期不计算logits_div和div_loss，不更新head_div参数
        # if self.head_div is not None and eval_training_finetuning(self.head_div_mode, self.in_finetuning):
        #     logits_div = self.head_div(last_token)  # only last token

        return {
            'fine_logits': fine_logits,
            'coarse_logits': coarse_logits,
            'div': logits_div,
            'fine_tokens': fine_tokens,
            'coarse_tokens': coarse_tokens
        }

    def forward(self, x, y):#此处y指的是segments而非labels
        fine_tokens, last_fine_token, coarse_tokens, last_coarse_token = self.forward_features(x, y)
        return self.forward_classifier(fine_tokens, last_fine_token, coarse_tokens, last_coarse_token)


def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False

# 获取真正的子类与父类索引的对应字典（层次字典）
def get_true_heirarchy_dict(dataset_name, class_names_to_idx):
    if 'cifar' in dataset_name:
        heirarchy_path = os.path.join("/home/yqn/dytox-hcast/iirc/metadata/iirc_cifar100_hierarchy.json")
    elif 'imagenet' in dataset_name:
        heirarchy_path = os.path.join("/home/yqn/dytox-hcast/iirc/metadata/iirc_imagenet_hierarchy_wnids.json")
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    with open(heirarchy_path) as f:
        class_hierarchy = json.load(f)

    superclass_to_subclasses = class_hierarchy['super_classes']

    # 对于cifar和imagenet_full, selected_supcls_to_subclses == superclass_to_subclasses;
    # 而对于imagenet_lite/subset, 读取层次json文件所得的superclass_to_subclass中包含了数据集(class_names_to_idx)没有的类，所以要做select
    selected_supcls_to_subclses = {
        superclass: [
            subclass_i for subclass_i in superclass_to_subclasses[superclass]
            if subclass_i in class_names_to_idx
        ]
        for superclass in superclass_to_subclasses
        if superclass in class_names_to_idx
    }

    selected_subclses_to_supcls = {subclass_i: superclass for superclass, subclasses in
                                selected_supcls_to_subclses.items()
                                for subclass_i in subclasses}

    sub_idx_to_super_idx = {class_names_to_idx[subclass]: class_names_to_idx[superclass] for
                            subclass, superclass in selected_subclses_to_supcls.items()}
    return selected_subclses_to_supcls, sub_idx_to_super_idx

