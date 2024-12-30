import cv2
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
img1 = cv2.imread('../target.jpg')
img2 = cv2.imread('../dataset/1.jpg')

# 转换为灰度图
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建 SIFT 对象
sift = cv2.SIFT_create()

# 检测关键点和计算描述子
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher()

# 使用 knnMatch 进行匹配
matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

# 应用比率测试来过滤错误匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.imshow(img3)
plt.show()