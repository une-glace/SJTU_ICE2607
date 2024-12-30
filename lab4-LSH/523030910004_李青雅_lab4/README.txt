压缩包结构：
523030910004_李青雅_lab3
-/codes
	-/LSH_color.py
	-/LSH_gradient.py
	-/LSH_gray.py
-/dataset
-/report
-/target.png
-/README

其中LSH_color.py为本次实验要求的代码，另外两个.py文件分别是以灰度和梯度作为信息实现的LSH算法。
运行时请确保codes和dataset、target在同一目录下，运行代码即可看到结果。
若需要修改投影集合I的值，请在120行到160行找到4行“g=”的注释掉的代码，修改成想要的值即可。
若要修改获取投影向量的方法，请在函数pre和lsh_search中使用不同的函数（Hamming或者hash）。