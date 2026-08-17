import torch
from timm.utils import dispatch_clip_grad

# 用于持续学习场景的梯度缩放器，封装了PyTorch的自动混合精度(AMP)功能(当disable_amp=True则不执行AMP)，用于解决精度丢失和梯度消失问题，提升训练效率
# 并提供了钩子机制(hook=True)支持模型参数更新前后的自定义操作。
class ContinualScaler:
    state_dict_key = "amp_scaler" # 用于 保存和加载状态字典 的 键名

    def __init__(self, disable_amp): # 根据disable_amp参数决定是否启用amp
        # 底层的 PyTorch梯度缩放器，负责实际的缩放操作
        self._scaler = torch.cuda.amp.GradScaler(enabled=not disable_amp)

    # 使ContinualScaler对象可直接调用，执行完整的梯度缩放流程
    def __call__(
        self, loss, optimizer, model_without_ddp, clip_grad=None, clip_mode='norm',
        parameters=None, create_graph=False,
        hook=True
    ):
        # 调用预处理步骤
        self.pre_step(loss, optimizer, parameters, create_graph, clip_grad, clip_mode)
        # 调用后处理步骤
        self.post_step(optimizer, model_without_ddp, hook)

    # 预处理步骤，包括损失缩放、反向传播和梯度裁剪
    def pre_step(self, loss, optimizer, parameters=None, create_graph=False, clip_grad=None, clip_mode='norm'):
        # 对损失进行缩放并反向传播
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # 反缩放优化器分配参数的梯度
        self._scaler.unscale_(optimizer)
        # 如果设置了梯度裁剪，则执行裁剪，用于缓解梯度爆炸
        if clip_grad is not None:
            assert parameters is not None
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

    # 后处理步骤，执行优化器更新并触发前后钩子函数
    def post_step(self, optimizer, model_without_ddp, hook=True):
        # 如果启用钩子且模型有前置更新钩子方法，则调用
        if hook and hasattr(model_without_ddp, 'hook_before_update'):
            model_without_ddp.hook_before_update()

        # 执行优化器步骤
        self._scaler.step(optimizer)

        # 如果启用钩子且模型有后置更新钩子方法，则调用
        if hook and hasattr(model_without_ddp, 'hook_after_update'):
            model_without_ddp.hook_after_update()

        # 更新缩放器
        self.update()

    # 更新内部缩放因子
    def update(self):
        self._scaler.update()

    # 获取缩放器的当前状态，用于保存检查点
    def state_dict(self):
        return self._scaler.state_dict()

    # 从状态字典恢复缩放器状态，用于恢复训练
    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
