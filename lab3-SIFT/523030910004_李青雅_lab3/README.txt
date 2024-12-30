压缩包结构：
-/523030910004_李青雅_lab3
	-/dataset(包含5张示例图片)
	-/codes
		-/exercise1.py(手动实现SIFT算法)
		-/exercise2.py(OpenCV实现SIFT算法)
	-/outputs
		-/exercise1(手动实现SIFT算法与原图的匹配结果)
		-/exercise2(OpenCV实现SIFT算法与原图的匹配结果)
	-/README
	-/report
	-/target(需要被对比的目标图片)


在手动实现SIFT算法中：
如果需要调整关键点的个数，请修改第12行中threshold的大小；threshold的值越小，关键点越多，反之同理。
如果需要调整匹配直线的条数，请修改141行内积和的阈值；阈值越大，直线条数越少，反之同理。