import os
from PIL import Image

# 设置图片库路径
image_dir = r"E:\ICE2607\lab5-PyTorch_CNN\CNN\piclib"  
output_dir = os.path.join(image_dir, "renamed_images")  # 新文件夹存储重命名的图片

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历文件夹并重命名
def rename_images(image_dir, output_dir):
    # 获取图片文件列表
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort()  # 可选，按文件名排序
    count = 1  # 从1开始编号

    for file in files:
        file_path = os.path.join(image_dir, file)
        
        # 打开图片并保存为 .png 格式
        try:
            img = Image.open(file_path)
            new_name = f"{count}.png"  # 重命名为1, 2, 3...的格式
            new_path = os.path.join(output_dir, new_name)

            img.save(new_path, format="PNG")
            print(f"Renamed and converted: {file} -> {new_name}")
            count += 1  # 编号递增
        except Exception as e:
            print(f"Failed to process {file}: {e}")

rename_images(image_dir, output_dir)
print("All files have been renamed and saved as .png!")
