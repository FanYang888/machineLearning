import math
import os
import sys
import json

import numpy as np

import torch.nn as nn
from torch.optim import lr_scheduler

from torchvision import transforms, datasets
from tqdm import tqdm

import os


import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from modelinceptionecarelub import densenet121, load_state_dict

import matplotlib.pyplot as plt



class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，8
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        # 计算测试样本的总数
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('真实类别')
        plt.ylabel('预测类别')


        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

def main():
    epochsArr = []
    accArr = []
    lossArr = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 对训练测试图片数据进行图像预处理
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "densenet_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=8)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = densenet121()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./densenet121-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'),False)
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.classifier.in_features
    net.classifier = nn.Linear(in_channel, 8)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    # params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=0.001)
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
    #
    # pg = [p for p in net.parameters() if p.requires_grad]
    # # 优化器
    # optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #
    # # lf = lambda x: ((1 + math.cos(x * math.pi / 50)) / 2) * (1 - 0.1) + 0.1  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 更新学习率

    #
    # pg = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=0.01, momentum=0.9, weight_decay=0,nesterov=True)
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #
    # lf = lambda x: ((1 + math.cos(x * math.pi / 50)) / 2) * (1 - 0.1) + 0.1 # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 更新学习率

    epochs = 50
    best_acc = 0.0
    save_path = 'end-all-delay0.010.0001SSSSSS.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # scheduler.step()

        class_indict = {
        "0": "\u4e94\u5cf0\u6bdb\u5c16",
        "1": "\u63ba\u504720%",
        "2": "\u63ba\u504740%",
        "3": "\u63ba\u504760%",
        "4": "\u63ba\u504780%",
        "5": "\u6709\u673a\u6bdb\u5c16",
        "6": "\u91c7\u82b1\u6bdb\u5c16",
        "7": "\u9ad8\u5c71\u6bdb\u5c16"
}
        label = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=8, labels=label)

        # validate

        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            net.eval()
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                confusion.update(predict_y.cpu().numpy(), val_labels.cpu().numpy())

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        epochsArr.append(epoch + 1)
        accArr.append(val_accurate)
        lossArr.append(running_loss / train_steps)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('accArr：' + str(accArr))
    print('loss：' + str(lossArr))
    print('Finished Training')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    confusion.plot()

    confusion.summary()
    plt.plot(epochsArr, accArr, color='b', label='accuracy')  # r表示红色
    plt.plot(epochsArr, lossArr, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色
    #####非必须内容#########
    plt.xlabel('epochs')  # x轴表示
    plt.ylabel('y label')  # y轴表示
    plt.title("")  # 图标标题表示
    plt.legend()  # 每条折线的label显示
    #######################
    plt.savefig('test.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片


if __name__ == '__main__':
    main()
