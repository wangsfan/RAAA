# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import os
import math
from typing import Iterable, Optional, Dict

import torch
from timm.data import Mixup
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy
from torch.nn import functional as F
from torch.utils import data

import continual.utils_iirc as utils
import metrics
from continual import factory_iirc_hcast
from continual.losses_iirc import DistillationLoss
from continual.pod import pod_loss
from iirc.lifelong_dataset.torch_dataset import Dataset
from lifelong_methods.utils import transform_labels_names_to_vector

CE = SoftTargetCrossEntropy()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    dataset_train: Dataset, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None,
                    sam: torch.optim.Optimizer = None,
                    loader_memory=None,
                    pod=None, pod_scales=[1], finetuning=False):
    """Code is a bit ugly to handle SAM, sorry! :upside_down_face:"""
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 10 # log every 10 batches
    
    loader_train = factory_iirc_hcast.get_train_loaders(dataset_train, args, finetuning)
    
    if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
        loader_train.sampler.set_epoch(epoch)
    # 当前已见类别数
    num_seen_classes = sum(model_without_ddp.nb_classes_per_task)

    # metric_logger.log_every(...): a generator that iterates over data_loader and logs progress every print_freq batches
    for batch_index, minibatch in enumerate(metric_logger.log_every(loader_train, print_freq, header)):
        labels_names = list(zip(minibatch[1], minibatch[2]))
        labels = transform_labels_names_to_vector(
            labels_names, num_seen_classes, args.class_names_to_idx
        )

        images = minibatch[0].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if len(minibatch) > 4:
            in_buffer = minibatch[3].to(device, non_blocking=True)
        else:
            in_buffer = None

        segments = minibatch[-1].to(device, non_blocking=True)

        if batch_index == 0:
            print(f'Image size is {images.shape}.')

        optimizer.zero_grad()  # Reset gradients to avoid accumulation

        lam = None
        if mixup_fn is not None:
            samples, targets, lam = mixup_fn(samples, targets)

        if sam is not None and (args.sam_first == 'memory' and task_id > 0):
            # If you want to do the first step of SAM only on memory samples.
            x, y, _ = loader_memory.get()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
        else:
            # with...amp.autocast包裹损失计算的代码块，实现在forward函数中使用混合精度（若enabled==True）计算损失
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple, predictions = forward(images, labels, segments, in_buffer, model, teacher_model, model_without_ddp, criterion, lam, args)

        loss = sum(filter(lambda x: x is not None, loss_tuple)) # forward内部计算各个损失时已经分别加权，此处只需求和
        # 如果需要自定义一些额外损失，在模型的该方法中定义并计算，以dict形式返回。此处无
        internal_losses = model_without_ddp.get_internal_losses(loss)
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        if pod is not None and teacher_model is not None: # 对于CIFAR，pod is None，故不执行该分支
            if args.pod_scaling:
                nb_classes = sum(model.module.nb_classes_per_task)
                nb_new_classes = model.module.nb_classes_per_task[-1]
                pod_scaling = math.sqrt(nb_new_classes / nb_classes)
            else:
                pod_scaling = 1.0

            loss += pod_scaling * pod * compute_pod(
                model.module.feats, teacher_model.feats, pod_scales)

        check_loss(loss) # 验证损失值非NaN

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if sam is not None and args.look_sam_k > 0:
            # Look-sam only apply the costly sam estimation every k step.
            look_sam_update = False
            if batch_index % args.look_sam_k == 0:
                loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
                loss_scaler.update()
                sam.first_step()  # modify weights to worse neighbor
                optimizer.zero_grad()

                look_sam_update = True

                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)
                loss = sum(filter(lambda x: x is not None, loss_tuple))
                internal_losses = model_without_ddp.get_internal_losses(loss)
                for internal_loss_value in internal_losses.values():
                    loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step(look_sam_update=look_sam_update)
            loss_scaler.post_step(optimizer, model_without_ddp)
        elif sam is not None:
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scaler.update()
            sam.first_step()  # modify weights to worse neighbor
            optimizer.zero_grad()

            if args.sam_second == 'memory' and task_id > 0:
                x, y, _ = loader_memory.get()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
            else:
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)

            loss = sum(filter(lambda x: x is not None, loss_tuple))
            internal_losses = model_without_ddp.get_internal_losses(loss)
            for internal_loss_value in internal_losses.values():
                loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step()
            loss_scaler.post_step(optimizer, model_without_ddp)
        else:
            loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()  # Wait for GPU operations to finish (for timing/metrics)

        batch_size = images.shape[0]
        # Update metric logger with losses and learning rate，以及iirc下的评价指标
        # metric_logger.update_dict(internal_losses)
        metric_logger.meters['fine_loss'].update(loss_tuple[0], n=batch_size)
        metric_logger.meters['coarse_loss'].update(loss_tuple[1], n=batch_size)
        metric_logger.meters['loss'].update(loss_tuple[2], n=batch_size)  # Main loss，其实就是分类损失
        if loss_tuple[3] is not None:
            metric_logger.meters['kd'].update(loss_tuple[3], n=batch_size)  # Knowledge distillation loss
        if loss_tuple[4] is not None:
            metric_logger.meters['div'].update(loss_tuple[4], n=batch_size)   # Divergence loss
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # Current learning rate
        # 更新metric logger中的iirc评价指标：jaccard_sim, modified_jaccard_sim, strict_accuracy, recall
        labels = labels.bool()
        metric_logger.meters['jaccard_sim'].update(round(metrics.jaccard_sim(predictions, labels), 5), n=batch_size)
        metric_logger.meters['modified_jaccard'].update(round(metrics.modified_jaccard_sim(predictions, labels), 5), n=batch_size)
        metric_logger.meters['strict_acc'].update(round(metrics.strict_accuracy(predictions, labels), 5), n=batch_size)
        metric_logger.meters['recall'].update(round(metrics.recall(predictions, labels), 5), n=batch_size)
        if debug:
            print('Debug, only doing one epoch!')
            break  # Stop after first batch if debugging

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()  # Model-specific post-epoch logic (e.g., weight averaging)

    # Gather metrics across distributed processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # 返回整个epoch所有batches上平均指标（各项损失，学习率，iirc下的评价指标）
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


def forward(samples: torch.Tensor, targets: torch.Tensor, segments, in_buffer: Optional[torch.Tensor], model, teacher_model, model_without_ddp, criterion, lam, args):
    # targets即labels已经转为了向量形式（one-hot/two-hots)
    fine_output, coarse_output, div_output = None, None, None

    outputs = model(samples, segments)
    if isinstance(outputs, dict):
        fine_output = outputs['fine_logits']
        coarse_output = outputs['coarse_logits']
        div_output = outputs['div']
    else:
        main_output = outputs

    main_output = torch.zeros_like(targets).float().to(targets.device)
    main_output[:, model_without_ddp.up2cur_subclasses_mask] = fine_output
    main_output[:, ~model_without_ddp.up2cur_subclasses_mask] = coarse_output

    predictions = main_output.ge(0.0)

    #粗细类分类损失分开计算
    f_weight = sum(model_without_ddp.up2cur_subclasses_mask) / sum(model_without_ddp.nb_classes_per_task)
    fine_loss = criterion(fine_output, targets[:, model_without_ddp.up2cur_subclasses_mask]) if f_weight > 0 else 0.0
    coarse_loss = criterion(coarse_output, targets[:, ~model_without_ddp.up2cur_subclasses_mask])
    loss = f_weight * fine_loss + (1-f_weight) * coarse_loss
    # 一起计算
    # loss = criterion(main_output, targets)

    if teacher_model is not None:
        with torch.no_grad():
            main_output_old = None
            teacher_outputs = teacher_model(samples, segments)

        if isinstance(outputs, dict):
            fine_output_old = teacher_outputs['fine_logits']
            coarse_output_old = teacher_outputs['coarse_logits']
        else:
            main_output_old = teacher_outputs

    kd_loss = None
    if teacher_model is not None:
        logits_for_distil = main_output[:, :sum(teacher_model.nb_classes_per_task)]
        fine_logits_for_distil = logits_for_distil[:, teacher_model.up2cur_subclasses_mask]
        coarse_logits_for_distil = logits_for_distil[:, ~teacher_model.up2cur_subclasses_mask]

        main_output_old = torch.zeros_like(logits_for_distil).float().to(targets.device)
        main_output_old[:, teacher_model.up2cur_subclasses_mask] = fine_output_old
        main_output_old[:, ~teacher_model.up2cur_subclasses_mask] = coarse_output_old

        tau = args.distillation_tau

        kd_loss = 0.
        if args.auto_kd:
            # Knowledge distillation on the probabilities
            # I called that 'auto_kd' because the right factor is automatically
            # computed, by interpolation between the main loss and the KD loss.
            # This is strongly inspired by WA (CVPR 2020) --> https://arxiv.org/abs/1911.07053
            lbd = main_output_old.shape[1] / main_output.shape[1]
            loss = (1 - lbd) * loss
            kd_factor = lbd

            # 同分类损失也可以分开/一起蒸馏，此处同分类设定，选择分开
            f_weight_old = sum(teacher_model.up2cur_subclasses_mask) / sum(teacher_model.nb_classes_per_task)
            fine_kd_loss = criterion(fine_logits_for_distil, torch.sigmoid(fine_output_old / tau)) if f_weight_old > 0 else 0.0
            coarse_kd_loss = criterion(coarse_logits_for_distil, torch.sigmoid(coarse_output_old / tau))
            _kd_loss = f_weight_old * fine_kd_loss + (1-f_weight_old) * coarse_kd_loss

            kd_loss += kd_factor * _kd_loss
        elif args.kd > 0.:
            _kd_loss = F.kl_div(
                    F.log_softmax(logits_for_distil / tau, dim=1),
                    F.log_softmax(main_output_old / tau, dim=1),
                    reduction='mean',
                    log_target=True
            ) * (tau ** 2)
            kd_loss += args.kd * _kd_loss


    div_loss = None
    if div_output is not None:# 效果不太好，暂不采用不执行
        # For the divergence heads, we need to create new targets.
        # If a class belong to old tasks, it will be 0.
        # If a class belong to the new task, it will be a class id between
        # 1 (not 0!) and 'nb_class_in_new_task'.
        # When doing that with mixup, some trickery is needed. (see if lam is not None).
        nb_classes = main_output.shape[1]
        nb_new_classes = div_output.shape[1] - 1
        nb_old_classes = nb_classes - nb_new_classes

        if lam is not None:  # 'lam' is the interpolation Lambda of mixup
            # If using mixup / cutmix
            div_targets = torch.zeros_like(div_output)
            nb_classes = main_output.shape[1]
            nb_new_classes = div_output.shape[1] - 1
            nb_old_classes = nb_classes - nb_new_classes

            div_targets[:, 0] = targets[:, :nb_old_classes].sum(-1)
            div_targets[:, 1:] = targets[:, nb_old_classes:]
        else:
            div_targets = torch.zeros_like(div_output)
            div_targets[:, 1:] = targets[:, nb_old_classes:]
            old_samples = div_targets.sum(dim=1) == 0 # 其实等同于in_buffer
            div_targets[old_samples, 0] = 1

        div_loss = args.head_div * criterion(div_output, div_targets)

    return (fine_loss, coarse_loss, loss, kd_loss, div_loss), predictions


def compute_pod(feats, old_feats, scales):
    if len(feats[0].shape) == 3:
        # transformer archi and not cnn
        bs, nb_tokens, dim = feats[0].shape
        w = int(math.sqrt(nb_tokens))
        feats = [f.view(bs, w, w, dim) for f in feats]
        old_feats = [f.view(bs, w, w, dim) for f in old_feats]

    return pod_loss(feats, old_feats, scales)


@torch.no_grad()
def evaluate(data_loader, model, model_without_ddp, device, args, header='Test:'):
    # 由于iirc设定下可能的多标签预测，故只能使用BCE
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    metric_logger = utils.MetricLogger(delimiter="  ")

    # 当前已见类别数
    if args.distributed:
        num_seen_classes = sum(model.module.nb_classes_per_task)
        # num_seen_classes = sum(model_without_ddp.nb_classes_per_task)
    else:
        num_seen_classes = sum(model.nb_classes_per_task)

    # switch to evaluation mode
    model.eval()

    for minibatch in metric_logger.log_every(data_loader, 10, header):
        labels_names = list(zip(minibatch[1], minibatch[2]))
        labels = transform_labels_names_to_vector(
            labels_names, num_seen_classes, args.class_names_to_idx
        )
        images = minibatch[0].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if len(minibatch) > 4:
            in_buffer = minibatch[3].to(device, non_blocking=True)
        else:
            in_buffer = None
        segments = minibatch[-1].to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images, segments)
            if isinstance(outputs, dict):
                fine_output = outputs['fine_logits']
                coarse_output = outputs['coarse_logits']

            output = torch.zeros_like(labels).float().to(labels.device)
            output[:, model_without_ddp.up2cur_subclasses_mask] = fine_output
            output[:, ~model_without_ddp.up2cur_subclasses_mask] = coarse_output

            loss = criterion(output, labels)

        predictions = output.ge(0.0)

        batch_size = images.shape[0]
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        # 更新metric logger中的iirc评价指标：jaccard_sim, modified_jaccard_sim, strict_accuracy, recall
        labels = labels.bool()
        metric_logger.meters['jaccard_sim'].update(metrics.jaccard_sim(predictions, labels), n=batch_size)
        metric_logger.meters['modified_jaccard'].update(metrics.modified_jaccard_sim(predictions, labels), n=batch_size)
        metric_logger.meters['strict_acc'].update(metrics.strict_accuracy(predictions, labels), n=batch_size)
        metric_logger.meters['recall'].update(metrics.recall(predictions, labels), n=batch_size)

        # predictions中预测出的类别对应的索引（预测结果可能不止一个类别）
        # labels中值为1的类别对应的索引
        # predicted_indices_batch = [torch.nonzero(pred_row, as_tuple=True)[0] for pred_row in predictions.cpu()]
        # label_indices_batch = [torch.nonzero(label_row, as_tuple=True)[0] for label_row in labels.cpu()]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {'loss': metric_logger.loss.global_avg}, {k: meter.global_avg for k, meter in metric_logger.meters.items() if k != 'loss'}

# phase==0 训练阶段的评估；==1 微调阶段的评估
def eval_and_log_in(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, n_parameters, device, dataset_test: Dataset, train_stats, log_store, log_path,
                 model_log, skipped_task=False, phase=0):
    # # 评估前保存（更新/覆盖）模型断点文件ckpt
    # # 同一任务内不同阶段保存的断点文件同名，所以会覆盖
    # if args.output_dir:
    #     if os.path.isdir(args.resume):
    #         checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
    #     else:
    #         checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
    #     for checkpoint_path in checkpoint_paths:
    #         if skipped_task:
    #             continue
    #
    #         utils.save_on_master({
    #             'datasets': {dataset_type: datasets[dataset_type].dataset_state_dict() for dataset_type in datasets.keys()},
    #             'buffer': buffer.buffer_state_dict(),
    #             'model': model_without_ddp.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'lr_scheduler': lr_scheduler.state_dict(),
    #             'epoch': epoch,
    #             'task_id': task_id,
    #             'scaler': loss_scaler.state_dict(),
    #             'args': args,
    #         }, checkpoint_path)
# def save_model(save_file: str, config: Dict, metadata: Dict, model: Optional['BaseMethod'] = None,
#                buffer: Optional['BufferBase'] = None, datasets: Optional[Dict[str, Dataset]] = None,
#                **kwargs) -> None:
#     """
#     Saves the experiment configuration and the state dicts of the model, buffer, datasets, plus any additional data
#
#     Args:
#         save_file (str): The checkpointing file path
#         config (Dict): The config of the experiment
#         metadata (Dict): The metadata of the experiment
#         model (Optional[BaseMethod]): The Method object (subclass of BaseMethod) for which the state dict should
#             be saved (Default: None)
#         buffer (Optional[BufferBase]): The buffer object for which the state dict should be saved (Default: None)
#         datasets (Optional[Dict[str, Dataset]]): The different dataset splits for which the state dict should be
#             saved (Default: None)
#         **kwargs: Any additional key value pairs that need to be saved
#     """
#     dicts = {}
#     dicts['config'] = config
#     dicts['metadata'] = metadata
#     if model is not None:
#         dicts['method_state_dict'] = model.method_state_dict()
#     if buffer is not None:
#         dicts['buffer_state_dict'] = buffer.buffer_state_dict()
#     if datasets is not None:
#         dicts['datasets_state_dict'] = {dataset_type: datasets[dataset_type].dataset_state_dict()
#                                         for dataset_type in datasets.keys()}
#     for key, value in kwargs.items():
#         dicts[key] = value
#     torch.save(dicts, save_file)
    assert dataset_test.complete_information_mode is True
    test_stats = {}

    dataset_test.choose_task(task_id)
    loader_test = factory_iirc_hcast.get_val_loaders(dataset_test, args)

    test_stats_loss, test_stats_metrics = evaluate(loader_test, model, model_without_ddp, device, args)

    test_stats.update(
        {f"task_{task_id}_test_{list(test_stats_loss.keys())[0]}": list(test_stats_loss.values())[0]})
    for metric, value in test_stats_metrics.items():
        test_stats.update({f"task_{task_id}_test_{metric}": round(value, 5)})

    log_stats = {'task': task_id,
                'phase': 'TR' if phase == 0 else 'FT',
                'epoch': epoch,
                'n_parameters': n_parameters,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **test_stats
                }

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({**log_stats, **model_log}) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

# t_endpoint==0 有微调且训练后微调前这一结束节点的全面评估；==1 任务结尾的结束节点的全面评估
def eval_and_log_end(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                     epoch, task_id, loss_scaler, n_parameters, device,
                     ave_test_m_jaccard_dict, ave_test_strict_acc_dict,
                     dataset_test: Dataset, train_stats, log_store, log_path,
                     model_log, skipped_task=False, t_endpoint=0):
    assert dataset_test.complete_information_mode is True

    test_stats_multi = {}
    # 为任务结尾，则要在各任务测试集和增量任务测试集上做评估（否则仅需在当前任务测试集上做评估）
    for test_t_id in range(task_id + 1):
        dataset_test.choose_task(test_t_id)
        loader_test = factory_iirc_hcast.get_val_loaders(dataset_test, args)

        if test_t_id == task_id:
            test_stats_loss, test_stats_metrics = evaluate(loader_test, model, model_without_ddp, device, args, header=f"Test_{test_t_id}:")
            # 其实就是task_id当前任务的测试集损失
            test_stats_multi.update(
                {f"task_{task_id}_test_{list(test_stats_loss.keys())[0]}": list(test_stats_loss.values())[0]})
        else:
            _, test_stats_metrics = evaluate(loader_test, model, model_without_ddp, device, args, header=f"Test_{test_t_id}:")

        for metric, value in test_stats_metrics.items():
            test_stats_multi.update({f"task_{test_t_id}_test_{metric}": round(value,5)})

    dataset_test.load_tasks_up_to(task_id)
    loader_test = factory_iirc_hcast.get_val_loaders(dataset_test, args)
    _, test_stats_metrics = evaluate(loader_test, model, model_without_ddp, device, args, header=f"Test_upto{task_id}:")

    for metric, value in test_stats_metrics.items():
        test_stats_multi.update({f"average_test_{metric}": round(value,5)})

    if ave_test_m_jaccard_dict is not None and ave_test_strict_acc_dict is not None: # 若为None则结果不完整，无需记录
        ave_test_m_jaccard_dict.update({f"task{task_id}": test_stats_metrics['modified_jaccard']}) # 保存精度更高的形式便于后期准确计算Ave
        ave_test_strict_acc_dict.update({f"task{task_id}": test_stats_metrics['strict_acc']})

    if t_endpoint==0: # 有微调且训练后微调前的全面评估结果
        assert args.finetuning
        print(f"* endpoint: after_TR_before_FT \n* {test_stats_multi}")
    elif t_endpoint==1: # 任务结尾的全面评估结果
        print(f"* endpoint: TASK_END \n* {test_stats_multi}")
    else:
        raise ValueError(f"Unknown endpoint {t_endpoint}")

    log_stats = {'task': task_id,
                't_endpoint': 'after_TR_before_FT' if t_endpoint == 0 else 'TASK_END',
                **{f'train_{k}': v for k, v in train_stats.items()},
                **test_stats_multi
                }

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({**log_stats, **model_log}) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
