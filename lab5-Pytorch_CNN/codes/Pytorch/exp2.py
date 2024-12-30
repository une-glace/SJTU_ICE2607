# SJTU EE208

'''Train CIFAR-10 with PyTorch.'''
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models import resnet20

# 检查是否有GPU支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

start_epoch = 0
end_epoch = 7
lr = 0.1

# Data pre-processing, DO NOT MODIFY
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

classes = ("airplane", "automobile", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck")

# Model
print('==> Building model..')
model = resnet20().to(device)  # 将模型移动到GPU
# If you want to restore training (instead of training from beginning),
# you can continue training based on previously-saved models
# by uncommenting the following two lines.
# Do not forget to modify start_epoch and end_epoch.
# restore_model_path = 'pretrained/ckpt_4_acc_63.320000.pth'
# model.load_state_dict(torch.load(restore_model_path)['net'])

# A better method to calculate loss
criterion = nn.CrossEntropyLoss()
# 使用SGD优化器
#optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
# 使用AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

# Mixup 数据增强
######################################################################
def mixup_data(x, y, alpha=1.0):
    '''Mixup 数据增强'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Mixup 损失函数'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
##########################################################################

# 使用Mixup数据增强的训练
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU
        
        # 使用 Mixup 数据增强
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.4)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 使用 Mixup 损失函数
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # 计算训练精度（以原始标签为基准，仅作为参考）
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).sum().item() + 
                    (1 - lam) * predicted.eq(targets_b).sum().item())
        
        print('Epoch [%d] Batch [%d/%d] Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
              % (epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1),
                 100. * correct / total, correct, total))


# # 普通训练
# def train(epoch):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         # The outputs are of size [128x10].
#         # 128 is the number of images fed into the model 
#         # (yes, we feed a certain number of images into the model at the same time, 
#         # instead of one by one)
#         # For each image, its output is of length 10.
#         # Index i of the highest number suggests that the prediction is classes[i].
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#         print('Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: %.3f%% (%d/%d)'
#               % (epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1),
#                  100. * correct / total, correct, total))


def test(epoch):
    print('==> Testing...')
    model.eval()
    with torch.no_grad():
        ##### TODO: calc the test accuracy #####
        # Hint: You do not have to update model parameters.
        #       Just get the outputs and count the correct predictions.
        #       You can turn to `train` function for help.
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU
            # optimizer.zero_grad()：优化器进行初始化
            optimizer.zero_grad()
            # inputs是x，model是function，outputs是f(x)
            outputs = model(inputs)
            # 损失（loss）：神经网络输出和目标之间的距离
            loss = criterion(outputs, targets)
            # train_loss是每次测试的损失
            train_loss += loss.item()
            # 记录输出的最大值（最好的匹配结果）
            _, predicted = outputs.max(1)
            # 记录目标总数
            total += targets.size(0)
            # 记录目标正确匹配数
            correct += predicted.eq(targets).sum().item()
            with open('result.txt', 'a') as f:
                f.write('TEST::Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: %.3f%% (%d/%d)\n'% (epoch, batch_idx + 1, len(testloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                # print('TEST::Epoch [%d] Batch [%d/%d] Loss: %.3f | Traininig Acc: %.3f%% (%d/%d)'% (epoch, batch_idx + 1, len(testloader), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = 1.0 * correct / total
        ########################################
    # Save checkpoint.
    print('Test Acc: %f' % acc)
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_%d_acc_%f.pth' % (epoch, acc))

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
for epoch in range(start_epoch, end_epoch + 1):
    train(epoch)
    test(epoch)
    scheduler.step()

start_epoch = 8
end_epoch = 19
lr = 0.01
# 使用SGD优化器
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
# 使用AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
for epoch in range(start_epoch, end_epoch + 1):
    train(epoch)
    test(epoch)
    scheduler.step()