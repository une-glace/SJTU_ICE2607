# SJTU EE208


# 使用ResNet50模型

# import time
# import os
# import matplotlib.pyplot as plt
# from PIL import Image

# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets.folder import default_loader

# from torchvision.models import resnet50, ResNet50_Weights

# print('Load model: ResNet50')
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# trans = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize,
# ])

# print('Prepare image data!')
# test_image = default_loader('white_dog.png')
# input_image = trans(test_image)
# input_image = torch.unsqueeze(input_image, 0)

# # resnet50的特征提取
# def features(x):
#     x = model.conv1(x)
#     x = model.bn1(x)
#     x = model.relu(x)
#     x = model.maxpool(x)
#     x = model.layer1(x)
#     x = model.layer2(x)
#     x = model.layer3(x)
#     x = model.layer4(x)
#     x = model.avgpool(x)

#     return x


# print('Extract features!')
# start = time.time()
# image_feature = features(input_image)
# image_feature = image_feature.detach().numpy()
# print('Time for extracting features: {:.2f}'.format(time.time() - start))


# print('Save features!')
# np.save('features.npy', image_feature)


# def dist(feature1,feature2):
#     # 需要归一化，否则容易过大
#     dis, t1, t2 = 0, 0, 0
#     for i in feature1[0]:
#         t1 += i[0][0]**2
#     t1 = t1**0.5
#     feature11 = [i[0][0]*1.0/t1 for i in feature1[0]]
#     for i in feature2[0]:
#         t2 += i[0][0]**2
#     t2 = t2**0.5
#     feature22 = [i[0][0]*1.0/t2 for i in feature2[0]]
#     for i in range(len(feature11)):
#         dis += (feature11[i]-feature22[i])**2
#     dis = dis**0.5
#     return dis

# def angle(feature1,feature2):
#     cnt, t1, t2 = 0, 0, 0
#     for i in feature1[0]:
#         t1 += i[0][0]**2
#     t1 = t1**0.5
#     feature11 = [i[0][0] for i in feature1[0]]
#     for i in feature2[0]:
#         t2 += i[0][0]**2
#     t2 = t2**0.5
#     feature22 = [i[0][0]for i in feature2[0]]
#     for i in range(len(feature22)):
#         cnt += feature11[i]*feature22[i]
#     cnt /= t1
#     cnt /= t2
#     return cnt

# # 展示与目标图片最相似的图片
# def show_similar_images(target_path, similar_images):
#     """
#     显示目标图片以及与之最相似的图片
#     :param target_path: 目标图片路径
#     :param similar_images: 与目标图片相似的图片列表，每个元素为 [距离, 文件名]
#     """
#     # 创建画布
#     num_images = len(similar_images) + 1  # 包括目标图片
#     plt.figure(figsize=(15, 5))

#     # 显示相似图片
#     for idx, (_, filename) in enumerate(similar_images):
#         plt.subplot(1, num_images, idx + 2)
#         similar_img = Image.open(os.path.join('piclib', filename))
#         plt.imshow(similar_img)
#         plt.title(f"Similar {idx + 1}")
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# distance = []
# # 调用函数去计算不同的欧氏距离情况比较
# start = time.time()
# for i in range(1,210):
#     filename = str(i) + '.png'
#     print('Prepare image data:  '+ filename + "  !")
#     # 此处读入i.png的图片，然后进行与前面待匹配图像一样的操作，故可略
#     temp_image = default_loader(os.path.join('piclib',filename))
#     test_image = trans(temp_image)
#     test_image = torch.unsqueeze(test_image, 0)

#     test_feature = features(test_image)
#     test_feature = test_feature.detach().numpy()    

#     # 欧氏距离内存下的为具体数值以及文件名
#     distance.append([dist(image_feature,test_feature),filename])
#     # 角度下的为具体数值及文件名
#     # distance.append([angle(image_feature,test_feature),filename])
    
# print('Time for extracting features: {:.2f}'.format(time.time() - start))

# for i in range(len(distance)):
#     for j in range(len(distance)):
#         # 如果是dist，则为<；如果是角度 则为>
#         if(distance[i][0]<distance[j][0]):
#             distance[i], distance[j] = distance[j], distance[i]
# for i in range(5):
#     print(distance[i])


# # 调用展示函数
# target_image_path = 'white_dog.png'
# most_similar_images = distance[:5]  # 取出距离最近的5张图片
# # 显示目标图片
# target_img = Image.open(target_image_path)
# plt.imshow(target_img)
# plt.title("Target Image")
# plt.axis("off")
# plt.show()
# show_similar_images(target_image_path, most_similar_images)




# 使用VGG16模型

import time
import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

from torchvision.models import vgg16, VGG16_Weights

print('Load model: VGG16')
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

print('Prepare image data!')
test_image = default_loader('white_dog.png')
input_image = trans(test_image)
input_image = torch.unsqueeze(input_image, 0)

def features(x):
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


print('Extract features!')
start = time.time()
image_feature = features(input_image)
image_feature = image_feature.detach().numpy()
print('Time for extracting features: {:.2f}'.format(time.time() - start))


print('Save features!')
np.save('features.npy', image_feature)


def dist(feature1, feature2):
    # 需要归一化，否则容易过大
    t1 = np.linalg.norm(feature1)
    t2 = np.linalg.norm(feature2)
    feature11 = feature1 / t1
    feature22 = feature2 / t2
    dis = np.linalg.norm(feature11 - feature22)
    return dis

def angle(feature1, feature2):
    t1 = np.linalg.norm(feature1)
    t2 = np.linalg.norm(feature2)
    # 将特征展平为一维
    feature11 = feature1.flatten() / t1
    feature22 = feature2.flatten() / t2
    # 计算点积
    cnt = np.dot(feature11, feature22)
    return cnt


# 展示与目标图片最相似的图片
def show_similar_images(target_path, similar_images):
    """
    显示目标图片以及与之最相似的图片
    :param target_path: 目标图片路径
    :param similar_images: 与目标图片相似的图片列表，每个元素为 [距离, 文件名]
    """
    # 创建画布
    num_images = len(similar_images) + 1  # 包括目标图片
    plt.figure(figsize=(15, 5))

    # 显示相似图片
    for idx, (_, filename) in enumerate(similar_images):
        plt.subplot(1, num_images, idx + 2)
        similar_img = Image.open(os.path.join('piclib', filename))
        plt.imshow(similar_img)
        plt.title(f"Similar {idx + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


distance = []
# 调用函数去计算不同的欧氏距离情况比较
start = time.time()
for i in range(1,210):
    filename = str(i) + '.png'
    print('Prepare image data:  '+ filename + "  !")
    # 此处读入i.png的图片，然后进行与前面待匹配图像一样的操作，故可略
    temp_image = default_loader(os.path.join('piclib',filename))
    test_image = trans(temp_image)
    test_image = torch.unsqueeze(test_image, 0)

    test_feature = features(test_image)
    test_feature = test_feature.detach().numpy()    

    # 欧氏距离内存下的为具体数值以及文件名
    # distance.append([dist(image_feature,test_feature),filename])
    # 角度下的为具体数值及文件名
    distance.append([angle(image_feature,test_feature),filename])
    
print('Time for extracting features: {:.2f}'.format(time.time() - start))

for i in range(len(distance)):
    for j in range(len(distance)):
        # 如果是dist，则为<；如果是角度 则为>
        if(distance[i][0]>distance[j][0]):
            distance[i], distance[j] = distance[j], distance[i]
for i in range(5):
    print(distance[i])


# 调用展示函数
target_image_path = 'white_dog.png'
most_similar_images = distance[:5]  # 取出距离最近的5张图片
# 显示目标图片
target_img = Image.open(target_image_path)
plt.imshow(target_img)
plt.title("Target Image")
plt.axis("off")
plt.show()
show_similar_images(target_image_path, most_similar_images)