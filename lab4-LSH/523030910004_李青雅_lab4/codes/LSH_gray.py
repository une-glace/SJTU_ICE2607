import cv2
from matplotlib import pyplot as plt
import time
# 首先获取图像对应的特征向量
# 通过颜色直方图计算相应的特征向量
def color(filename):
    # 读入图（目标的特征向量）
    in_img = cv2.imread(filename+'.jpg')
    image = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    # 创建0~255的灰度值像素点数
    # 为了与12配套，故此处设为264 = 12 * 22
    gray = [0]*264
    tot  = 0
    for i in range(0,len(image)):
        for j in range(0,len(image[0])):
            # 直接将image[i][j]点的灰度像素值加到我们的gray上
            gray[image[i][j]] += 1
            tot += 1
    # 建立具体比例，存入p中
    p = []
    for i in range(12):
        cnt = 0
        for j in range(22):
            cnt += gray[i*22+j]
        p.append(cnt*1.0/tot)
    # 计算出0、1、2类型的特征向量
    for i in range(len(p)):
        if p[i] < 0.08:
            p[i] = 0
        elif p[i] < 0.16:
            p[i] = 1
        else:
            p[i] = 2
    return p

# 计算Hamming码来获取投影，其中g是需要被投影的方向
def Hamming(p,g):
    # 由于p是12维的向量，构造v(p)需要24维的数组
    lsh = []
    v = [0] * 24
    for i in range(12):
        # 当前位置只有00,10,11三种情况，第一二位分别对应2*i,2*i+1
        if p[i] == 1:
            v[2*i] = 1
        if p[i] == 2:
            v[2*i] = 1
            v[2*i+1] = 1
    # eg.v(p)=001011100011
    # 对于上述 p，它在{1,3,7,8}上的投影为(0,1,1,0)
    for i in g:
        lsh.append(v[i-1])
    return lsh


# 哈希函数计算lsh
def hash(p,g):
    # I是Hash桶
    I  = [[] for i in range(12)]
    # print(I)
    lsh = []
    # I|i表示I中范围在(i-1)*C+1~i*C中的坐标：
    for i in g:
        # print((i-1)//2)
        I[(i-1)//2].append(i)
    for i in range(12):
        if I[i] != []:
            for Ii in I[i]:
                if Ii-2*i <= p[i]:
                    lsh.append(1)
                else:
                    lsh.append(0)
    return lsh


# 对搜索库中的图像（共40张，均以i.jpg为文件名）建立索引
def pre(g): 
    Hash, Hash_id, character = [], [], []
    for i in range(1,41):
        filename ='dataset/'+str(i)
        p = color(filename)
        character.append(p)
        # 可用I的方式获取投影
        # 也可以用汉明法获取投影即lsh = Hamming(p,g)
        lsh = hash(p,g)
        # 需要注意的是，由于循环的时候文件用的是1.jpg->40.jpg
        # 然而数组append只能是从0开始，因而在具体实现的时候需要-1，同样回答了输出的+1原因
        try:
            Hash_id[Hash.index(lsh)].append(i - 1)
        except:
            Hash.append(lsh)
            Hash_id.append([i - 1])
    return Hash, Hash_id, character


# 利用lsh检索图片，进行图片索引的查找
def lsh_search(g, Hash, Hash_id, character):
    result = []
    p = color('target')
    lsh = hash(p,g)
    # 判断该图片的特征向量是否合理
    if lsh not in Hash:
        return result
    ID = Hash.index(lsh)
    # 这里的NN法可以直接判断向量是否相等，这样直接且高效
    # print(Hash_id[ID])
    for i in Hash_id[ID]:
        if character[i] == p:
            result.append('dataset/'+str(i+1)+'.jpg')
    return result

# 不使用LSH，直接利用NN法进行搜索
def nn_search(character):
    result = []
    p = color('target')
    for i in range(0,40):
        # 直接枚举
        if character[i] == p:
            result.append('dataset/'+str(i+1)+'.jpg')
    return result


# g = [1,7,8,10,15,20]
g = [1,7,8,10,20]
# g = [1,7,10,15,20]
# g = [1,3,7,8,17,19,21]
# g = color('target')

# 首先进行预处理，为所有待查图像建立相应索引
Hash, Hash_id, character = pre(g)
# print(character)

start = time.time()
print("When using lsh_search......")
result = lsh_search(g, Hash, Hash_id, character)
end = time.time()
print("The efficiency of lsh_search is {}ms".format((end-start)*1000))
if result == []:
    print("No matching picture!")
else:
    print("There is(are) {} result(s), and they are(it is):".format(len(result)))
    for i in result:   
        print('\t' + i)


start = time.time()
print("When using nn_search......")
result = nn_search(character)
end = time.time()
print("The efficiency of nn_search is {}ms".format((end-start)*1000))
if result == []:
    print("No matching picture!")
else:
    print("There is(are) {} result(s), and they are(it is):".format(len(result)))
    for i in result:   
        print('\t' + i)