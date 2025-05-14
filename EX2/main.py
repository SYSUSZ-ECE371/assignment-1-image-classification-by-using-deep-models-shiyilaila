import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights  # 导入权重枚举类
from torch.utils.data import random_split
import os
import time
import copy

# 设置数据集目录路径
data_dir = 'EX2/flower_dataset'

# 定义数据增强和归一化转换操作
# 这些操作将应用于训练和验证阶段的图像数据
data_transforms = transforms.Compose([
    # 随机裁剪图像并调整为224×224像素大小
    transforms.RandomResizedCrop(224),
    # 随机水平翻转图像，增加数据多样性
    transforms.RandomHorizontalFlip(),
    # 随机旋转图像±10度
    transforms.RandomRotation(10),
    # 随机调整图像亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # 以5%的概率将图像转换为灰度图
    transforms.RandomGrayscale(p=0.05),
    # 将图像转换为PyTorch张量
    transforms.ToTensor(),
    # 使用ImageNet数据集的均值和标准差进行归一化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 定义函数用于初始化模型，避免重复加载
def initialize_model(num_classes):
    # 使用预训练的ResNet18模型，权重基于在ImageNet数据集上的训练结果
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 修改模型的最后一层（全连接层）以适应我们的分类任务
    # 获取原始全连接层的输入特征数
    num_ftrs = model.fc.in_features
    # 替换原始全连接层为新的全连接层，输出维度为类别数
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# 定义训练模型的函数
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    # 记录训练开始时间
    since = time.time()

    # 保存最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 确定运行设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练主循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        # 每个轮次都有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            # 初始化损失和正确预测数
            running_loss = 0.0
            running_corrects = 0

            # 迭代数据批次
            for inputs, labels in dataloaders[phase]:
                # 将输入和标签移至指定设备
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 只有在训练阶段才跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播
                    outputs = model(inputs)
                    # 获取预测类别
                    _, preds = torch.max(outputs, 1)
                    # 计算损失
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才进行反向传播和优化
                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数

                # 统计损失和正确预测数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 训练阶段更新学习率
            if phase == 'train':
                scheduler.step()

            # 计算并打印本轮的损失和准确率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存验证集上准确率最高的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 创建保存目录
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)
                # 保存最佳模型权重
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print()

    # 打印训练总耗时和最佳验证准确率
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 主程序入口
if __name__ == '__main__':
    # 加载完整数据集并应用转换
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # 划分数据集为训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 准备数据加载器字典和数据集大小字典
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    num_classes = len(full_dataset.classes)

    # 初始化模型
    model = initialize_model(num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义学习率调度器，每7个轮次将学习率降低为原来的0.1倍
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 确定运行设备并将模型移至该设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 训练模型
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25)