import argparse, json
import datetime
import os
import numpy as np
import logging
import torch, random

from mpc_server import *
from mpc_client import *
import models
from AES import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from authorize_module import *
def read_dataset():
    data_X, data_Y = [], []

    with open("/Users/ziyichen/Desktop/pythonProject6/data/breast.csv") as fin:
        for line in fin:
            data = line.split(',')
            data_X.append([float(e) for e in data[:-1]])
            if int(data[-1]) == 1:
                data_Y.append(1)
            else:
                data_Y.append(-1)

    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    print("one_num: ", np.sum(data_Y == 1), ", minus_one_num: ", np.sum(data_Y == -1))

    idx = np.arange(data_X.shape[0])
    np.random.shuffle(idx)

    train_size = int(data_X.shape[0] * 0.8)

    train_x = data_X[idx[:train_size]]
    train_y = data_Y[idx[:train_size]]

    eval_x = data_X[idx[train_size:]]
    eval_y = data_Y[idx[train_size:]]

    return (train_x, train_y), (eval_x, eval_y)

def authorize():
    ski = generate_aes_key()
    return ski



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open('/Users/ziyichen/Desktop/pythonProject6/utils/conf.json', 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = read_dataset()

    print(train_datasets[0].shape, train_datasets[1].shape)

    print(eval_datasets[0].shape, eval_datasets[1].shape)

    # 多个server 每个server计算一部分数据
    servers = []
    # server = Server(conf, eval_datasets)
    for s in range(conf["servers"] - 1):
        server = Server(conf, eval_datasets, conf["mpc_feature_num"])
        servers.append(server)
    servers.append(Server(conf, eval_datasets, conf["mpc_feature_num"] + conf["mpc_feature_num_remain"]))

    ski = authorize()
    clients = []
    train_size = train_datasets[0].shape[0]
    per_client_size = int(train_size / conf["no_models"])
    for c in range(conf["no_models"]):
        clients.append(Client(c, conf, servers[0].get_public_key(), servers,
                              train_datasets[0][c * per_client_size: (c + 1) * per_client_size],
                              train_datasets[1][c * per_client_size: (c + 1) * per_client_size], ski))

    # print(server.global_model.weights)
    # 身份认证
    authorize_module = authorize_module(ski)
    for client in clients:
        authorize_module.check_authority(client.sign(), client.get_rsa_public_key())
        if authorize_module.check_authority(client.sign(), client.get_rsa_public_key()):
            print("client ", client.id, " pass authority check")
            continue
        else:
            print("authorization check failed")
            exit(-1)

    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, conf["k"])
        encrypted_weights_list = []
        for server in servers:
            server.global_model.encrypt_weights = models.encrypt_vector(server.public_key,
                                                                        models.decrypt_vector(server.private_key,
                                                                                         server.global_model.encrypt_weights))
            for w in server.global_model.encrypt_weights:
                encrypted_weights_list.append(w)

        weight_accumulators = []
        for s in range(conf["servers"] - 1):
            weight_accumulators.append([servers[s].public_key.encrypt(0.0)] * conf["mpc_feature_num"])
        weight_accumulators.append([servers[conf["servers"] - 1].public_key.encrypt(0.0)] *
                                   (conf["mpc_feature_num"] + conf["mpc_feature_num_remain"]))

        for c in candidates:
            #         # print(models.decrypt_vector(Server.private_key, server.global_model.encrypt_weights))
            diff = c.local_train(encrypted_weights_list)
            w_num = 0
            for w in weight_accumulators:
                for i in range(len(w)):
                    j = i + w_num * conf["mpc_feature_num"]
                    w[i] = w[i] + diff[j]
                w_num = w_num + 1

        encrypted_weights_list_test = []
        for s in range(conf["servers"]):
            servers[s].model_aggregate(weight_accumulators[s])
            for w in servers[s].global_model.encrypt_weights:
                encrypted_weights_list_test.append(w)

        acc = model_eval(conf, eval_datasets[0], eval_datasets[1], encrypted_weights_list_test)

        print("Epoch %d, acc: %f\n" % (e, acc))
