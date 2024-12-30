import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 更改工作目录到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 让直方图中有中文!
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 负号

def canny_edge_detection(image, low_threshold, high_threshold):
    """
    手动实现 Canny 边缘检测
    1. 转灰度
    2. 使用 Sobel 算子计算梯度
    3. 计算梯度幅值和方向
    4. 非极大值抑制
    5. 双阈值检测和边缘连接
    """
    # 1. 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 使用 Sobel 算子计算梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # 梯度方向（角度）
    direction = (direction + 180) % 180  # 转为正角度
    
    # 3. 非极大值抑制
    nms = np.zeros_like(magnitude)
    rows, cols = gray.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            angle = direction[r, c]
            angle = angle % 180  # 将角度限制在0到180度之间

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                before = magnitude[r, c - 1]
                after = magnitude[r, c + 1]
            elif 22.5 <= angle < 67.5:
                before = magnitude[r - 1, c + 1]
                after = magnitude[r + 1, c - 1]
            elif 67.5 <= angle < 112.5:
                before = magnitude[r - 1, c]
                after = magnitude[r + 1, c]
            else:
                before = magnitude[r - 1, c - 1]
                after = magnitude[r + 1, c + 1]

            # 线性插值
            weight = (angle % 45) / 45.0
            interpolated_value = before * (1 - weight) + after * weight

            if magnitude[r, c] >= interpolated_value:
                nms[r, c] = magnitude[r, c]
    
    # 4. 双阈值检测
    edges = np.zeros_like(nms, dtype=np.uint8)
    strong = np.int32(nms > high_threshold)
    weak = np.int32((nms >= low_threshold) & (nms <= high_threshold))
    edges[strong == 1] = 255  # 强边缘
    edges[weak == 1] = 50  # 弱边缘（暂时标记）
    
    # 5. 弱边缘连接
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if edges[r, c] == 50:  # 弱边缘
                if 255 in edges[r - 1:r + 2, c - 1:c + 2]:  # 如果邻域中有强边缘
                    edges[r, c] = 255
                else:
                    edges[r, c] = 0  # 否则丢弃

    return edges

# 读取图像
image = cv2.imread("../../dataset/3.jpg")  # 替换为您的图片路径

# 设置低阈值和高阈值
low_threshold = 70
high_threshold = 200

# 手动实现 Canny 边缘检测
manual_edges = canny_edge_detection(image, low_threshold, high_threshold)

# 使用 OpenCV 自带的 Canny 方法
opencv_edges = cv2.Canny(image, low_threshold, high_threshold)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Manual Canny")
plt.imshow(manual_edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("OpenCV Canny")
plt.imshow(opencv_edges, cmap="gray")
plt.axis("off")

plt.show()
