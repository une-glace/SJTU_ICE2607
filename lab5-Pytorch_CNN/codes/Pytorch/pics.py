import re
import matplotlib.pyplot as plt

# 初始化存储数据的列表
epochs = []
test_accs = []

# 从 result.txt 文件中读取测试准确率
with open('result.txt', 'r') as f:
    for line in f:
        # 匹配 TEST:: 的日志行
        if "TEST::" in line:
            # 提取 Epoch 和 Test Accuracy
            match = re.search(r'Epoch \[(\d+)] .* Traininig Acc: ([\d\.]+)', line)
            if match:
                epoch = int(match.group(1))
                test_acc = float(match.group(2))
                epochs.append(epoch)
                test_accs.append(test_acc)

# 检查提取的数据
print("Epochs:", epochs)
print("Test Accuracies:", test_accs)

# 绘制测试准确率变化趋势
plt.figure(figsize=(8, 6))
plt.plot(epochs, test_accs, marker='o', label="Test Accuracy")
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.xticks(epochs)  # 显示每个 Epoch 的刻度
plt.grid(False)
plt.legend()
plt.show()