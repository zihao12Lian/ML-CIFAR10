import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF  # 引入高斯噪声
from torchvision.models import resnet18

# ---------------- 图像预处理 ----------------

# 自定义高斯噪声（也可以用外部库如 albumentations）
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# 数据增强 pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),                      # 随机旋转 ±15°
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), # 随机裁剪缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),                  # 10% 概率转灰度
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- 数据加载 ----------------

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
classes = train_dataset.classes

# ---------------- 定义辅助模块 ----------------

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MixedPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, alpha=0.5):
        super(MixedPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * self.max_pool(x) + (1 - self.alpha) * self.avg_pool(x)
#mish 激活函数
def convert_relu_to_mish(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, Mish())
        else:
            convert_relu_to_mish(module)
    return model

# ---------------- 构建模型 ----------------

model = resnet18(num_classes=10)
model.maxpool = MixedPool2d(kernel_size=3, stride=2, alpha=0.7)
model = convert_relu_to_mish(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张GPU进行训练")
    model = nn.DataParallel(model)

# ---------------- 初始化训练参数 ----------------

os.makedirs("models", exist_ok=True)
best_model_path = 'models/best_model.pth'

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

train_losses = []
train_accuracies = []
test_accuracies = []
learning_rates = []

best_accuracy = 0.0

# ---------------- 开始训练 ----------------

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
    print(f"→ 当前学习率: {scheduler.get_last_lr()[0]:.6f}")

    train_losses.append(running_loss)
    learning_rates.append(scheduler.get_last_lr()[0])
    scheduler.step()

    # 训练集准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    # 测试集准确率
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 新的最佳模型已保存，准确率: {best_accuracy:.2f}%")

# ---------------- 绘图保存 ----------------

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, marker='o', label='Train Acc')
plt.plot(epochs, test_accuracies, marker='o', label='Test Acc')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs, learning_rates, marker='o', color='orange')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
