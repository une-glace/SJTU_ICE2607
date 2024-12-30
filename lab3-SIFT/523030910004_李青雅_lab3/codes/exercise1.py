import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from math import pi
import os

# 更改工作目录到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Harris角点检测
def detect_harris_corners(image, block_size=2, ksize=3, k=0.04, threshold=0.001):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    corners = np.where(dst > threshold * dst.max())
    return list(zip(corners[1], corners[0])), dst

# 关键点检测
def get_kp(img, max_corners=100, threshold=0.01, min_distance=10):
    corners, dst = detect_harris_corners(img, threshold=threshold)
    kp = np.zeros((len(corners), 2), dtype=np.int16)
    for i, corner in enumerate(corners):
        kp[i][0] = corner[0]
        kp[i][1] = corner[1]
    
    # 非极大值抑制
    keypoints = []
    for y, x in corners:
        if 0 <= y < dst.shape[0] and 0 <= x < dst.shape[1]:  # 边界检查
            local_max = np.max(dst[max(0, y-1):min(y+2, dst.shape[0]), max(0, x-1):min(x+2, dst.shape[1])])
            if dst[y, x] == local_max:
                keypoints.append((x, y))
    
    # 按响应值排序并限制关键点数量
    keypoints = sorted(keypoints, key=lambda pt: dst[pt[1], pt[0]], reverse=True)
    keypoints = keypoints[:max_corners]
    
    # 应用最小距离参数
    final_keypoints = []
    for i, (x, y) in enumerate(keypoints):
        if all(np.linalg.norm(np.array([x, y]) - np.array([kx, ky])) >= min_distance for kx, ky in final_keypoints):
            final_keypoints.append((x, y))
    
    kp = np.array(final_keypoints, dtype=np.int16)
    return kp

# 计算梯度幅值
def get_amp(img):
    height, wide = img.shape
    img_ca = np.array(img, dtype=float)
    amply = np.zeros((height, wide))
    for i in range(1, height-1):
        for j in range(1, wide-1):
            amply[i][j] = ((img_ca[i][j+1] - img_ca[i][j-1])**2 + (img_ca[i+1][j] - img_ca[i-1][j])**2)**0.5
    return amply

# 计算梯度方向
def get_the(img):
    height, wide = img.shape
    img_ca = np.array(img, dtype=float)
    theta = np.zeros((height, wide))
    for i in range(1, height-1):
        for j in range(1, wide-1):
            theta[i][j] = math.atan2(img_ca[i][j+1] - img_ca[i][j-1], img_ca[i+1][j] - img_ca[i-1][j])
    return theta

# 计算主方向
def main_direct(amp, the, img_gray, sigma):
    dir = []
    r = int(4.5 * sigma)
    h, w = img_gray.shape
    kp = get_kp(img_gray)
    for k in kp:
        x, y = k
        vote = [0 for i in range(36)]
        for i in range(max(0, y-r), min(y+r+1, h)):
            for j in range(max(0, x-r), min(x+r+1, w)):
                index = int((the[i][j] + pi) * 18.0 / pi - 1)
                vote[index] += amp[i][j]
        t = 0
        for i in range(len(vote)):
            if vote[i] > vote[t]:
                t = i
        dir.append(t * pi / 18 + pi / 36 - pi)
    return dir

# 计算描述子
def get_xy_after_spinning(x, y, th):
    x1 = math.cos(th) * x - math.sin(th) * y
    y1 = math.sin(th) * x + math.cos(th) * y
    return [x1, y1]
def get_xy2(x1, y1, sigma):
    return [(x1 / (3 * sigma)) + 2, (y1 / (3 * sigma)) + 2]
def get_descriptors(amp, the, img_gray):
    descriptors = []
    h, w = img_gray.shape
    sigma = (h + w) / 690
    radius = int(sigma * 15 / (2**0.5))
    dir = main_direct(amp, the, img_gray, sigma)
    kp = get_kp(img_gray)
    for k in range(len(kp)):
        x0, y0 = kp[k]
        descriptor = np.zeros((4, 4, 8))
        for i in range(max(0, y0-radius), min(y0+radius+1, h)):
            for j in range(max(0, x0-radius), min(x0+radius+1, w)):
                th = the[i][j]
                th = int((th + pi - 0.000001) // (pi / 4))
                x = j - x0
                y = i - y0
                trans1 = get_xy_after_spinning(x, y, dir[k])
                trans2 = get_xy2(trans1[0], trans1[1], sigma)
                x2 = trans2[0] - 0.5
                y2 = trans2[1] - 0.5
                index_x = math.floor(x2)
                index_y = math.floor(y2)
                if index_x >= 0 and index_y >= 0 and index_x < 4 and index_y < 4:
                    descriptor[index_x][index_y][th] += (index_x + 1 - x2) * (index_y + 1 - y2)
                if index_x >= 0 and index_y < 3 and index_x < 4 and index_y >= -1:
                    descriptor[index_x][index_y+1][th] += (index_x + 1 - x2) * (y2 - index_y)
                if index_x < 3 and index_y >= 0 and index_x >= -1 and index_y < 4:
                    descriptor[index_x+1][index_y][th] += (x2 - index_x) * (index_y + 1 - y2)
                if index_x < 3 and index_y < 3 and index_x >= -1 and index_y >= -1:
                    descriptor[index_x+1][index_y+1][th] += (x2 - index_x) * (y2 - index_y)
        ret = np.linalg.norm(descriptor, axis=None)
        if ret != 0:
            descriptor = descriptor * (1 / ret)
        descriptors.append(descriptor)
    return descriptors

# 图像匹配
def match(des1, des2):
    num = 0
    mat = []
    mat_n = []
    mat_index = []
    for i in range(len(des1)):
        for j in range(len(des2)):
            mat_n.append(np.sum(des1[i] * des2[j]))
            mat_index.append([i, j])
            if np.sum(des1[i] * des2[j]) > 0.5:
                num += 1
    print(num)
    arr = np.array(mat_n)
    idx = arr.argsort()[-num:][::-1]
    for i in range(num):
        mat.append(mat_index[idx[i]])
    return [num > 5, mat]

# sift算法
def sift(img):
    kp = get_kp(img)
    amply = get_amp(img)
    theta = get_the(img)
    des = get_descriptors(amply, theta, img)
    return [kp, des]

target = cv2.imread('../target.jpg')
tar_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
tar_gray = cv2.GaussianBlur(tar_gray, (3, 3), 0.6)
keytar, destar = sift(tar_gray)

# 两张图竖着放
for i in range(1, 6):
    img = cv2.imread('../dataset/%d.jpg' % i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (3, 3), 0.6)
    key, des = sift(gauss)
    print('matches of %d.jpg:' % i)
    flag, mat = match(des, destar)
    w1, h1 = gauss.shape
    w2, h2 = tar_gray.shape
    newimg = np.zeros((w1 + w2, max(h1, h2), 3), dtype=np.int16)
    for m in range(w1):
        for n in range(h1):
            for s in range(3):
                newimg[m][n][s] = img[m][n][2-s]
    for m in range(w2):
        for n in range(h2):
            for s in range(3):
                newimg[w1 + m][n][s] = target[m][n][2-s]
    for j in range(len(mat)):
        kp0 = key[mat[j][0]]
        kp1 = []
        kp1.append(keytar[mat[j][1]][0])
        kp1.append(keytar[mat[j][1]][1] + w1)
        bgr = np.random.randint(0, 255, 3, dtype=np.int32)
        cv2.line(newimg, kp0, kp1, (int(bgr[0]), int(bgr[1]), int(bgr[2])), 2)
        
    plt.imshow(newimg)
    plt.show()


# # 两张图横着放
# for i in range(1, 6):
#     img = cv2.imread('E:/ICE2607/lab3-SIFT/dataset/%d.jpg' % i)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gauss = cv2.GaussianBlur(gray, (3, 3), 0.6)
#     key, des = sift(gauss)
#     print('matches of %d.jpg:' % i)
#     flag, mat = match(des, destar)
#     w1, h1 = gauss.shape
#     w2, h2 = tar_gray.shape
#     newimg = np.zeros((max(w1, w2), h1 + h2, 3), dtype=np.int16)
#     for m in range(w1):
#         for n in range(h1):
#             for s in range(3):
#                 newimg[m][n][s] = img[m][n][2-s]
#     for m in range(w2):
#         for n in range(h2):
#             for s in range(3):
#                 newimg[m][h1 + n][s] = target[m][n][2-s]
#     for j in range(len(mat)):
#         kp0 = key[mat[j][0]]
#         kp1 = []
#         kp1.append(keytar[mat[j][1]][0] + h1)
#         kp1.append(keytar[mat[j][1]][1])
#         bgr = np.random.randint(0, 255, 3, dtype=np.int32)
#         cv2.line(newimg, kp0, kp1, (int(bgr[0]), int(bgr[1]), int(bgr[2])), 2)
        
#     plt.imshow(newimg)
#     plt.show()