# ex1_config.py
# 基于ResNet50的ImageNet预训练配置
# 使用MMClassification库中预定义的ResNet50配置作为基础配置
_base_ = 'mmpretrain::resnet/resnet50_8xb32_in1k.py'

# 完整的模型配置
model = dict(
    # 模型类型为图像分类器
    type='ImageClassifier',
    # 主干网络配置
    backbone=dict(
        type='ResNet',  # 使用ResNet架构
        depth=50,       # ResNet50，包含50层
        num_stages=4,   # 使用4个阶段的网络结构
        out_indices=(3,),  # 只输出最后一个阶段的特征
        style='pytorch',  # 使用PyTorch风格的网络结构
    ),
    # 颈部网络配置，用于特征处理
    neck=dict(type='GlobalAveragePooling'),  # 使用全局平均池化
    # 头部网络配置，用于分类
    head=dict(
        type='LinearClsHead',  # 使用线性分类头
        num_classes=5,         # 分类类别数量为5（对应5类花卉）
        in_channels=2048,      # 输入特征维度，ResNet50最后一层输出为2048维
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # 使用交叉熵损失函数
        topk=(1,),             # 评估指标：计算Top-1准确率
    )
)

# 数据集配置
data_root = './data/flower_dataset'  # 数据集根目录

# 训练数据加载器配置
train_dataloader = dict(
    dataset=dict(
        type='ImageNet',  # 数据集类型，按ImageNet格式组织（图像文件夹+标注文件）
        data_root=data_root,  # 数据集根路径
        ann_file='train.txt',  # 训练集标注文件，记录图像路径和标签
        data_prefix='train',   # 训练图像所在文件夹
    ),
    batch_size=32,  # 每个批次加载32张图像
    num_workers=4,  # 使用4个工作进程加载数据
)

# 验证数据加载器配置
val_dataloader = dict(
    dataset=dict(
        type='ImageNet',  # 数据集类型
        data_root=data_root,  # 数据集根路径
        ann_file='val.txt',    # 验证集标注文件
        data_prefix='val',     # 验证图像所在文件夹
    ),
    batch_size=32,  # 验证批次大小
    num_workers=4,  # 数据加载工作进程数
)

# 训练配置
train_cfg = dict(
    max_epochs=15,  # 训练总轮次，即整个数据集将被训练15次
)

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',   # 优化器类型：随机梯度下降
        lr=0.0008,     # 学习率，控制参数更新的步长
        momentum=0.92, # 动量参数，帮助加速收敛并减少震荡
        weight_decay=0.00015  # 权重衰减，用于正则化，防止过拟合
    )
)

# 预训练权重路径（用于模型微调）
# 指定从ImageNet预训练的ResNet50模型权重文件加载
# 模型将基于这些权重进行微调，以适应花卉分类任务
load_from = './checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'