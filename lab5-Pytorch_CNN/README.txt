压缩包结构：
523030910004_李青雅_lab5
-/codes
	-/Pytorch
		-/exp2.py
		-/models.py
		-/pics.py（用于绘制test acc变化趋势）
		-/final.pth
		-/result.txt
	-/CNN
		-/piclib（图片库）
		-/Dataset
		-/dealing_photos.py（用于将图片全部改为.png格式的脚本，可以自行导入图片后，再运行这个脚本）
		-/extract_feature.py
		-/features.npy（目标图像的特征）
		-/white_dog.png（目标图像，可进行替换）
-/report
-/README

由于写完了报告之后才发现助教补发的Dataset，但此时修改报告工作量太大，所以我还是用的自己大约200张的图像库
助教如果测试的时候需要用到Dataset，只需将Dataset中的50张图片复制到piclib中，再将extract_feature.py中第256行的png改为jpg即可