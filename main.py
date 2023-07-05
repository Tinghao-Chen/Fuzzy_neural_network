import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from FFDN import FusedFuzzyDeepNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
epoch_num = 5
input_dim = 28 * 28
fuzz_dim = 100
num_class = 10
batch_size = 128
learning_rate = 10e-5

#数据集操作
mnist_dataset = datasets.MNIST('', train=True,
                               transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                               download=True)
train_set, valid_set = random_split(mnist_dataset, [50000, 10000])

# Split validation set into validation and test sets
valid_set, test_set = random_split(valid_set, [2000, 8000])
#数据集载入
train_set_loader = DataLoader(train_set, batch_size=batch_size)
valid_set_loader = DataLoader(valid_set, batch_size=batch_size)

#模型载入
model = FusedFuzzyDeepNet(input_dim, fuzz_dim, num_class).to(device)
# 最小验证loss定义
min_valid_loss = np.inf

#计算交叉熵损失
criterion = nn.CrossEntropyLoss()
# 创建优化器对象，实现随机梯度下降算法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    # 对每个epoch，训练loss初始化为0
    train_loss = 0.0
    for data, labels in train_set_loader:
        # 训练集加载后，将数据与标签传入gpu
        data, labels = data.to(device), labels.to(device)
        # 优化器梯度归0
        optimizer.zero_grad()
        # 执行model(data)将会调用FusedFuzzyDeepNet模型的forward()方法，将data作为输入传递给模型。
        # 模型会根据定义的网络结构和权重参数进行前向传播，生成输出结果target
        # 其中target的形状应该是(batch_size, num_class)，表示每个样本在各个类别上的预测概率分布。
        target = model(data)
        # loss是交叉熵计算结果

        loss = criterion(target, labels)
        # 将loss后向传播，优化器再进行学习参数迭代，更新模型
        loss.backward()
        optimizer.step()
        # loss.item是将损失值进行标量化，计算的是一个batch size的平均损失
        # data.size(0)返回输入数据data的第一个维度的大小，即批次大小（batch size）。
        train_loss = loss.item() * data.size(0)

   valid_loss = 0.0
    # 将模型设置为评估模式,在训练阶段，模型会根据训练数据进行参数更新和学习，以提高模型的性能。
    # 而在评估阶段，模型主要用于对新样本进行预测，并评估模型的性能。
    model.eval()
    for data, labels in valid_set_loader:
        data, labels = data.to(device), labels.to(device)
        #
        target = model(data)
        loss = criterion(target, labels)
        valid_loss += loss.item() * data.size(0)

    train_loss /= len(train_set_loader.dataset)
    valid_loss /= len(valid_set_loader.dataset)
    print('Epoch: {:d} - training loss: {:.6f} - validation loss: {:.6f}'.format(epoch, train_loss, valid_loss))

# Set the model to evaluation mode
model.eval()
# Create a dataloader for the test dataset
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
# Variables for tracking test loss and accuracy
test_loss = 0.0
correct_predictions = 0
total_predictions = 0

# Disable gradient computation for testing
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)

        # Perform inference
        target = model(data)

        # Calculate the loss
        loss = criterion(target, labels)

        # Accumulate the test loss
        test_loss += loss.item() * data.size(0)

        # Calculate the predicted labels
        _, predicted_labels = torch.max(target, 1)

        # Count the number of correct predictions
        correct_predictions += (predicted_labels == labels).sum().item()

        # Count the total number of predictions
        total_predictions += labels.size(0)

# Calculate the average test loss and accuracy
average_test_loss = test_loss / len(test_set)
accuracy = correct_predictions / total_predictions

# Print the test results
print("Test Loss: {:.4f}".format(average_test_loss))
print("Accuracy: {:.2f}%".format(accuracy * 100))
