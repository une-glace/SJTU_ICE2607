import cv2
import os
from matplotlib import pyplot as plt
import matplotlib

# 更改工作目录到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 让直方图中有中文!
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 负号

# 确保 test1 文件夹存在
test_dir = "test1"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

def plot_color_ratio(image_path, title = "颜色比例直方图"):
    # 读取图像并转换为 RGB 格式
    img_bgr = cv2.imread(image_path)
    # 防止读取失败
    if img_bgr is None:
        print(f"无法读取图像: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 计算每个颜色通道的能量总和
    color_sum = [img_rgb[:, :, i].sum() for i in range(3)]
    total_energy = sum(color_sum)

    # 计算每个颜色通道的比例
    color_ratio = [energy / total_energy for energy in color_sum]

    # 绘制颜色比例直方图
    colors = ['红', '绿', '蓝']
    plt.bar(colors, color_ratio, color=['red', 'green', 'blue'])
    plt.xlabel("颜色通道")
    plt.ylabel("颜色比例")
    plt.title(title)
    # 设置 y 轴范围从 0 到 1
    plt.ylim(0, 1)  

    # 在每个柱状图上方显示具体的比例值
    for i, ratio in enumerate(color_ratio):
        plt.text(i, ratio + 0.01, f"{ratio:.2f}", ha='center', color='black')


image_paths = ["images/img1.jpg", "images/img2.jpg", "images/img3.jpg"]
for i, image_path in enumerate(image_paths, start=1):
    plot_color_ratio(image_path, title=f"图片 {i} 的颜色比例直方图")
    plt.savefig(f"{test_dir}/test1_{i}.png")
    plt.show()
    
