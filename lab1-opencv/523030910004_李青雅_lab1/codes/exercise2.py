import cv2
import os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

# 更改工作目录到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 让直方图中支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 负号

# 确保 test2 文件夹存在
test_dir = "test2"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

def plot_gray_ratio_filled(image_path, title="灰度比例直方图"):
    # 读取图像为灰度图
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"无法读取图像: {image_path}")
        return

    # 计算灰度直方图
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    # 归一化处理，使得直方图值代表比例
    hist_normalized = hist / hist.sum()

    # 绘制直方图并填充颜色
    plt.figure(figsize=(8, 6))
    x = range(256)  # 灰度值范围
    plt.fill_between(x, hist_normalized.flatten(), color='teal', alpha=1)  # 填充颜色，找了个很好看的水鸭色（

    plt.title(title)
    plt.xlabel("灰度值 (0-255)")
    plt.ylabel("像素比例")


def compute_gradient_histogram(image_path, bins=361, title="梯度强度直方图"):
    # 读取图像为灰度图
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 计算梯度 (Sobel算子)
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

    # 计算梯度强度
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 将梯度强度分桶统计
    max_magnitude = np.max(magnitude)
    hist, bin_edges = np.histogram(magnitude, bins=bins, range=(0, max_magnitude))

    # 归一化直方图
    hist_normalized = hist / hist.sum()

    # 绘制直方图
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], hist_normalized, width=(bin_edges[1] - bin_edges[0]), color='teal', alpha=1)
    plt.title(title)
    plt.xlabel("梯度强度值范围")
    plt.ylabel("像素比例")


image_paths = ["images/img1.jpg", "images/img2.jpg", "images/img3.jpg"]
for i, image_path in enumerate(image_paths, start=1):
    plot_gray_ratio_filled(image_path, title=f"图片 {i} 的灰度比例直方图")
    plt.savefig(f"{test_dir}/test2_1_{i}.png")
    plt.show()
for i, image_path in enumerate(image_paths, start=1):
    compute_gradient_histogram(image_path, title=f"图片 {i} 的梯度强度直方图")
    plt.savefig(f"{test_dir}/test2_2_{i}.png")
    plt.show()

