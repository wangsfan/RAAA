from functools import partial

import torch
from torch import nn

from cast_models import cast_deit_hier_iirc
from cast_models.modules_iirc import ConvStem
from continual import convit, dytox_hcast, samplers_iirc, vit
from continual.cnn import (InceptionV3, resnet18, resnet34, resnet50,
                           resnext50_32x4d, seresnet18, vgg16, vgg16_bn,
                           wide_resnet50_2, resnet18_scs, resnet18_scs_max, resnet18_scs_avg, resnet_rebuffi)


def get_backbone(args):
    print(f"Creating model: {args.model}")
    if args.model == 'cast':
        # embed_layer采用ConvStem时patch_size只能为8（否则要改ConvStem模型），cifar数据集难以满足，故采用vit传统的PatchEmbed
        # 另外，depth分为[3,3,3,2]，即整体比常规vit少1层的原因就是因为使用了ConvStem，所以不使用的话就可以恢复？
        # model = cast_deit_hier_iirc.CAST(# cifar的patch_size设为4/2？
        #     patch_size=args.patch_size, embed_dim=args.embed_dim, num_clusters=args.num_clusters,
        #     depth=args.depth, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem,
        #     drop_rate=args.drop, drop_path_rate=args.drop_path)

        model = cast_deit_hier_iirc.CAST(# cifar的patch_size设为4/2？
            img_size=args.input_size, patch_size=args.patch_size, embed_dim=args.embed_dim, num_clusters=args.num_clusters,
            depth=args.depth, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=args.drop, drop_path_rate=args.drop_path)

    elif args.model == 'vit':
        model = vit.VisionTransformer(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads
        )
    elif args.model == 'convit':
        model = convit.ConVit(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            local_up_to_layer=args.local_up_to_layer,
            locality_strength=args.locality_strength,
            class_attention=args.class_attention,
            ca_type='jointca' if args.joint_tokens else 'base',
            norm_layer=args.norm,
        )
    elif args.model == 'resnet18_scs': model = resnet18_scs()
    elif args.model == 'resnet18_scs_avg': model = resnet18_scs_max()
    elif args.model == 'resnet18_scs_max': model = resnet18_scs_avg()
    elif args.model == 'resnet18': model = resnet18()
    elif args.model == 'resnet34': model = resnet34()
    elif args.model == 'resnet50': model = resnet50()
    elif args.model == 'wide_resnet50': model = wide_resnet50_2()
    elif args.model == 'resnext50': model = resnext50_32x4d()
    elif args.model == 'seresnet18': model = seresnet18()
    elif args.model == 'inception3': model = InceptionV3()
    elif args.model == 'vgg16bn': model = vgg16_bn()
    elif args.model == 'vgg16': model = vgg16()
    elif args.model == 'rebuffi': model = resnet_rebuffi()
    else:
        raise NotImplementedError(f'Unknown backbone {args.model}')

    return model



def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = samplers_iirc.get_sampler(dataset_train, dataset_val, args)

    # 微调时（即finetuning==True 且 args.ft_no_sampling==True，实际CIFAR即是）sampler为None，shuffle采取默认值False，不打乱顺序
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None if (finetuning and args.ft_no_sampling) else sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(sampler_train) > args.batch_size,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, finetuning=False, batch_size=None, drop_last=True):
    batch_size = batch_size or args.batch_size

    sampler_train = samplers_iirc.get_train_sampler(dataset_train, args)

    # 微调时（即finetuning==True 且 args.ft_no_sampling==True，实际CIFAR即是）sampler为None，shuffle采取默认值False，不打乱顺序
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None if (finetuning and args.ft_no_sampling) else sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(sampler_train) > args.batch_size,
    )

    return loader_train


def get_val_loaders(dataset_val, args):
    sampler_val = samplers_iirc.get_val_sampler(dataset_val, args)

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return loader_val



class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return self.get()


def update_dytox(model_without_ddp, task_id, args):
    if task_id == 0:
        print(f'Creating DyTox!')
        model_without_ddp = dytox_hcast.DyTox(model_without_ddp, args)
    else:
        print(f'Updating ensemble, new embed dim {model_without_ddp.embed_dim}.')
        model_without_ddp.add_model(args.increment)

    return model_without_ddp
