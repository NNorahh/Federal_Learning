import models, torch, copy
import numpy as np
from mpc_server import Server
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import RSA
from base64 import decodebytes, encodebytes
from sha256rsa import *


class Client(object):

    def __init__(self, id, conf, public_key, servers, data_x, data_y, ski):
        self.id = id
        self.conf = conf

        self.public_key = public_key
        self.sha256rsa = RSAUtil(ski)
        self.ski = ski

        encrypted_weights_list = []
        for server in servers:
            # print(server.global_model.encrypt_weights)
            for w in server.global_model.encrypt_weights:
                encrypted_weights_list.append(w)

        # self.w =

        self.local_model = models.LR_Model(public_key=self.public_key, w_size=conf["feature_num"] + 1,
                                           w=encrypted_weights_list, encrypted=True)

        # print(type(self.local_model.encrypt_weights))
        self.data_x = data_x

        self.data_y = data_y

    def sign(self):
        return self.sha256rsa.sign(self.ski)

    def get_rsa_public_key(self):
        return self.sha256rsa.public_key

    def local_train(self, encrypted_weights_list):
        # encrypted_weights_list = []
        # for server in servers:
        #     # print(server.global_model.encrypt_weights)
        #     for w in server.global_model.encrypt_weights:
        #         encrypted_weights_list.append(w)
        # encrypted_weights_list = server.global_model.encrypt_weights for server in servers
        # print(len(encrypted_weights_list))
        origin_w = encrypted_weights_list
        # print(origin_w)
        self.local_model.set_encrypt_weights(origin_w)
        neg_one = self.public_key.encrypt(-1)
        for e in range(self.conf["local_epochs"]):
            print("start epoch ", e)
            # if e > 0 and e%2 == 0:
            #	print("re encrypt")
            #	self.local_model.encrypt_weights = Server.re_encrypt(self.local_model.encrypt_weights)

            idx = np.arange(self.data_x.shape[0])
            batch_idx = np.random.choice(idx, self.conf['batch_size'], replace=False)
            # print(batch_idx)

            x = self.data_x[batch_idx]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.data_y[batch_idx].reshape((-1, 1))

            # print((0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one).shape)

            # print(x.transpose().shape)

            # assert(False)

            batch_encrypted_grad = x.transpose() * (
                    0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
            encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]

            for j in range(len(self.local_model.encrypt_weights)):
                self.local_model.encrypt_weights[j] -= self.conf["lr"] * encrypted_grad[j]

        weight_accumulators = []
        # print(models.decrypt_vector(Server.private_key, weights))
        for j in range(len(self.local_model.encrypt_weights)):
            weight_accumulators.append(self.local_model.encrypt_weights[j] - origin_w[j])

        return weight_accumulators
