#本地端
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import dataloader


class Host:
    def __init__(self, cid, train_set, test_set, lr = 0.1):
        self.lr = lr  # 学习率
        self.cid = cid  # 客户端ID
        self.train_set = train_set
        self.test_sett = test_set
        self.train_loader = dataloader.DataLoader(dataset=train_set, batch_size=1000, shuffle=False)
        self.test_loader = dataloader.DataLoader(dataset=test_set, batch_size=1000, shuffle=False)


    def train_and_test_local(self):
        class NeuralNet(nn.Module):
            def __init__(self, input_num, hidden_num, output_num):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_num, hidden_num)  # 服从正态分布的权重w
                self.fc2 = nn.Linear(hidden_num, output_num)
                nn.init.normal_(self.fc1.weight)
                nn.init.normal_(self.fc2.weight)
                nn.init.constant_(self.fc1.bias, val=0)  # 初始化bias为0
                nn.init.constant_(self.fc2.bias, val=0)
                self.relu = nn.ReLU()  # Relu激励函数

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                y = self.fc2(x)
                return y

        epoches = 20  # 迭代20轮
        lr = 0.01  # 学习率，即步长
        input_num = 784
        hidden_num = 12
        output_num = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NeuralNet(input_num, hidden_num, output_num)
        model.to(device)
        loss_func = nn.CrossEntropyLoss()  # 损失函数的类型：交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化，也可以用SGD随机梯度下降法
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epoches):
            flag = 0
            for images, labels in self.train_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                output = model(images)

                loss = loss_func(output, labels)
                optimizer.zero_grad()
                loss.backward()  # 误差反向传播，计算参数更新值
                optimizer.step()  # 将参数更新值施加到net的parameters上

                # 以下两步可以看每轮损失函数具体的变化情况
                # if (flag + 1) % 10 == 0:
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
                flag += 1

        params = list(model.named_parameters())  # 获取模型参数

        # 测试，评估准确率
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            output = model(images)
            values, predicte = torch.max(output, 1)  # 0是每列的最大值，1是每行的最大值
            total += labels.size(0)
            # predicte == labels 返回每张图片的布尔类型
            correct += (predicte == labels).sum().item()
        print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
        return params

    def FL_train_and_test(self, com_para_fc1, com_para_fc2):
        class NeuralNet(nn.Module):
            def __init__(self, input_num, hidden_num, output_num, com_para_fc1, com_para_fc2):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_num, hidden_num)
                self.fc2 = nn.Linear(hidden_num, output_num)
                self.fc1.weight = Parameter(com_para_fc1)
                self.fc2.weight = Parameter(com_para_fc2)
                nn.init.constant_(self.fc1.bias, val=0)
                nn.init.constant_(self.fc2.bias, val=0)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                y = self.fc2(x)
                return y

        epoches = 20
        lr = 0.01
        input_num = 784
        hidden_num = 12
        output_num = 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NeuralNet(input_num, hidden_num, output_num, com_para_fc1, com_para_fc2)
        model.to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(epoches):
            flag = 0
            for images, labels in self.train_loader:
                # (images, labels) = data
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                output = model(images)

                loss = loss_func(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (flag + 1) % 10 == 0:
                # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoches, loss.item()))
                flag += 1
        params = list(model.named_parameters())  # get the index by debuging

        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            output = model(images)
            values, predicte = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicte == labels).sum().item()
        print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
        return params

