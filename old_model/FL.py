# import math
#
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.utils.data.dataloader as dataloader
# from torch.utils.data import Subset
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parameter import Parameter
# import matplotlib.pyplot as plt
#
# from Guest import Guest
# from Host import Host
#
# host_num = 3
#
# # 训练集
# train_set = torchvision.datasets.MNIST(
#     root="./data",
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True)
#
# test_set = torchvision.datasets.MNIST(
#     root="./data",
#     train=False,
#     transform=transforms.ToTensor(),
#     download=True)
# test_set = Subset(test_set, range(3000, 5000))
# test_loader = dataloader.DataLoader(dataset=test_set, shuffle=True)
#
# # 只对w做平均
#
#
# if __name__ == '__main__':
#
#     # train_data_loader = torch.utils.data.DataLoader(
#     #     dataset=train_set,
#     #     batch_size=64,
#     #     shuffle=True,
#     #     drop_last=True)
#     # images, labels = next(iter(train_data_loader))
#     # img = torchvision.utils.make_grid(images, padding=0)
#     # img = img.numpy().transpose(1, 2, 0)
#     # plt.imshow(img)
#     # plt.show()
#
#     # train_set_A = Subset(train_set, range(0, 1000))
#     # train_set_B = Subset(train_set, range(1000, 2000))
#     # train_set_C = Subset(train_set, range(2000, 3000))
#     # train_set_combine = []
#     # train_set_combine.append(train_set_A)
#     # train_set_combine.append(train_set_B)
#     # train_set_combine.append(train_set_C)
#
#     hosts = []
#     for i in range(host_num):
#         train_set_for_i = Subset(train_set, range(math.floor(60000/host_num)*i,
#                            math.floor(60000/host_num)*(i+1)))
#         # train_set_for_i = train_set_combine[i]
#         test_set = test_set
#         host = Host(cid=i, train_set=train_set_for_i, test_set=test_set)
#         hosts.append(host)
#
#     print('\033[31m'+'Start training model ABC at 1st time...'+'\033[0m')
#     #先本地训练
#     train_model_parameter = []
#     for host in hosts:
#         print(f'Training on client {host.cid}...')
#         trained_model = host.train_and_test_local()
#         # print(trained_model)
#         train_model_parameter.append(trained_model)
#
#     aggregator = Guest()
#     # print(para_A)
#     # 联邦后训练 只聚合w
#     for i in range(6):
#         print('\033[31m'+'The {} round to be federated!!!'.format(i + 1)+'\033[0m')
#         com_para_fc1, com_para_fc2 = aggregator.combine_params(train_model_parameter)
#         train_model_parameter = []
#         for host in hosts:
#             print(f'Training on client {host.cid}...')
#             trained_model = host.FL_train_and_test(com_para_fc1, com_para_fc2)
#             train_model_parameter.append(trained_model)
#
#
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import RSA
from base64 import decodebytes, encodebytes

def generate_key_pair():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def sign(plain_text, private_key):
    key = RSA.import_key(private_key)
    h = SHA256.new(plain_text.encode())
    signer = PKCS1_v1_5.new(key)
    signature = signer.sign(h)
    return encodebytes(signature).decode()

def verify(plain_text, signature, public_key):
    key = RSA.import_key(public_key)
    h = SHA256.new(plain_text.encode())
    verifier = PKCS1_v1_5.new(key)
    signature = decodebytes(signature.encode())
    return verifier.verify(h, signature)

if __name__ == "__main__":
    # 生成 RSA 密钥对
    private_key, public_key = generate_key_pair()

    # 明文
    plain_text = "hello world"

    # 私钥签名
    c = sign(plain_text, private_key)
    print("签名结果:", c)

    # 公钥验签
    print("验证结果:", verify(plain_text, c, public_key))
