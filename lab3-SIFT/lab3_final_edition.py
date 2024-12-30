import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import random
PI = math.pi
# 基本照搬上个lab的梯度图代码，返回梯度值以及所获得的的角度angle
def gradient(img):
    Ix, Iy = np.zeros((len(img), len(img[0]))), np.zeros((len(img), len(img[0])))
    M, angle = np.zeros((len(img), len(img[0]))), np.zeros((len(img), len(img[0])))
    for i in range(1, len(img) - 1):
        for j in range(1, len(img[0]) - 1):
            Ix[i][j] = int(img[i + 1][j]) - int(img[i - 1][j])
            Iy[i][j] = int(img[i][j + 1]) - int(img[i][j - 1])
            M[i][j] = (Ix[i][j]**2 + Iy[i][j]**2)**0.5
            angle[i][j] = np.arctan2(Iy[i][j], Ix[i][j])  # Fixed gradient angle calculation
    return M, angle

def sift(img):
    row, column = np.shape(img)
    
    # 调整特征点检测的参数以提高敏感性
    corners = cv2.goodFeaturesToTrack(img, maxCorners=400, qualityLevel=0.02, minDistance=10)
    corners = [np.int8(corner[0]) for corner in corners]

    img = cv2.GaussianBlur(img, (5, 5), 1)
    grad, angle = gradient(img)
    
    direction = []
    bins = (row + column) // 80
    for corner in corners:
        y, x = corner
        ccnt = [0] * 37
        for i in range(max(x - bins, 0), min(x + bins + 1, row)):
            for j in range(max(y - bins, 0), min(y + bins + 1, column)):
                weight = int((angle[i][j] + PI) * 1.0 / (PI / 18) + 1)
                weight = min(weight, 36)
                ccnt[weight] += grad[i][j]
        maxn = np.argmax(ccnt)
        direction.append((maxn * 1.0 / 18 - 1 - 1.0 / 36) * PI)

    def Featurelize(point, θ):
        def θ_(x, y):
            if (x < 0 or x >= row) or (y < 0 or y >= column):
                return 0
            tmp = angle[x][y] - θ
            return tmp + 2 * PI if tmp < 0 else tmp

        def Bilinear_Interpolation(x_1, y_1):
            x, y = int(x_1), int(y_1)
            dx1, dy1 = x_1 - x, y_1 - y
            dx2, dy2 = 1 - dx1, 1 - dy1
            θθ = θ_(x, y) * dx2 * dy2 + θ_(x + 1, y) * dx1 * dy2 + θ_(x, y + 1) * dx2 * dy1 + θ_(x + 1, y + 1) * dx1 * dy1
            return θθ

        y0, x0 = point
        Horizon = np.array([np.cos(θ), np.sin(θ)])
        Vertical = np.array([-np.sin(θ), np.cos(θ)])

        def cnt(x1, x2, y1, y2, signx, signy):
            count = [0] * 9
            for x in range(x1, x2):
                for y in range(y1, y2):
                    dp = [x * signx, y * signy]
                    p = Horizon * dp[0] + Vertical * dp[1]
                    weig = int((Bilinear_Interpolation(p[0] + x0, p[1] + y0)) // (PI / 4) + 1)
                    weig = min(weig, 8)
                    count[weig] += 1
            return count[1:]

        vector = []
        bins = (row + column) // 150
        for xsign in [-1, 1]:
            for ysign in [-1, 1]:
                vector += cnt(0, bins, 0, bins, xsign, ysign)
                vector += cnt(bins, bins * 2, 0, bins, xsign, ysign)
                vector += cnt(bins, bins * 2, bins, bins * 2, xsign, ysign)
                vector += cnt(0, bins, bins, bins * 2, xsign, ysign)
        return vector

    feature = []
    for i, corner in enumerate(corners):
        des = Featurelize(corner, direction[i])
        norm = sum(k * k for k in des) ** 0.5
        feature.append([k * 1.0 / norm for k in des])
    return feature, corners

def Merge(img1, img2):
    h1, w1, a = np.shape(img1)
    h2, w2, a = np.shape(img2)
    if h1 < h2:
        extra = np.zeros((h2 - h1, w1, 3), dtype=np.uint8)
        img1 = np.vstack([img1, extra])
    elif h1 > h2:
        extra = np.zeros((h1 - h2, w2, 3), dtype=np.uint8)
        img2 = np.vstack([img2, extra])
    return np.hstack([img1, img2])

if __name__ == "__main__":
    target0 = cv2.imread('E:/ICE2607/lab3-SIFT/target.jpg')
    imgpkp0 = cv2.imread('E:/ICE2607/lab3-SIFT/dataset/5.jpg')
    target = cv2.cvtColor(target0, cv2.COLOR_BGR2GRAY)
    imgpkp = cv2.cvtColor(imgpkp0, cv2.COLOR_BGR2GRAY)

    feature_target, corners_target = sift(target)
    feature_img, corners_img = sift(imgpkp)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(np.array(feature_target, dtype=np.float32), np.array(feature_img, dtype=np.float32), k=2)

    # 应用Lowe's ratio test过滤匹配
    good_matches = []
    ratio_thresh = 0.85
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 限制绘制的匹配数量
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

    # Convert corners to KeyPoint objects with float coordinates
    keypoints_target = [cv2.KeyPoint(float(c[0]), float(c[1]), 1) for c in corners_target]
    keypoints_img = [cv2.KeyPoint(float(c[0]), float(c[1]), 1) for c in corners_img]

    img_matches = cv2.drawMatches(
        target0, keypoints_target,
        imgpkp0, keypoints_img,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    outputpath = "output/matchOfImage2.jpg"
    cv2.imwrite(outputpath, img_matches)