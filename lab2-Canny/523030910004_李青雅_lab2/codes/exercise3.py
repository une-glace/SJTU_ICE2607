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
    使用 Canny 算子实现边缘检测
    1. 转灰度
    2. 使用 Canny 算子计算梯度
    3. 计算梯度幅值和方向
    4. 非极大值抑制
    5. 双阈值检测和边缘连接
    """
    # 1. 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 使用 Canny 算子计算梯度
    edges = cv2.Canny(gray, low_threshold, high_threshold)  # 这里的阈值可以根据需要调整
    
    # Canny 算子本身已经计算了边缘，因此不需要单独计算 magnitude 和 direction
    magnitude = edges
    direction = None  # Canny 算子不直接提供方向信息
        
    # 3. 非极大值抑制
    nms = np.zeros_like(magnitude)
    rows, cols = gray.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            angle = direction[r, c] if direction is not None else 0
            angle = angle % 180  # 将角度限制在0到180度之间
            
            # 根据梯度方向进行非极大值抑制
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[r, c-1], magnitude[r, c+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[r-1, c+1], magnitude[r+1, c-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[r-1, c], magnitude[r+1, c]]
            else:
                neighbors = [magnitude[r-1, c-1], magnitude[r+1, c+1]]
            
            if magnitude[r, c] >= max(neighbors):
                nms[r, c] = magnitude[r, c]
            else:
                nms[r, c] = 0
    
    # 4. 双阈值检测和边缘连接
    strong_edges = (nms > high_threshold).astype(np.uint8)
    weak_edges = ((nms >= low_threshold) & (nms <= high_threshold)).astype(np.uint8)
    
    edges = np.zeros_like(nms)
    edges[strong_edges == 1] = 255
    
    # 边缘连接
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if weak_edges[r, c] == 1:
                if np.any(strong_edges[r-1:r+2, c-1:c+2] == 1):
                    edges[r, c] = 255
    
    return edges

# 读取图像
image = cv2.imread("../../dataset/1.jpg")  # 替换为您的图片路径

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

plt.savefig(f"../outputs/exercise3/1/Canny.jpg")  # 保存结果图像
plt.show()