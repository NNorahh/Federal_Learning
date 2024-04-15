import models, torch

import paillier

import numpy as np

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)


def model_eval(conf, eval_x, eval_y, global_model_encrypt_weights):
    total_loss = 0.0
    correct = 0
    dataset_size = 0

    batch_num = int(eval_x.shape[0] / conf["batch_size"])

    global_model_weights = models.decrypt_vector(private_key, global_model_encrypt_weights)
    # print(global_model_weights)

    for batch_id in range(batch_num):
        x = eval_x[batch_id * conf["batch_size"]: (batch_id + 1) * conf["batch_size"]]
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        y = eval_y[batch_id * conf["batch_size"]: (batch_id + 1) * conf["batch_size"]].reshape(
            (-1, 1))

        dataset_size += x.shape[0]

        wxs = x.dot(global_model_weights)

        pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]

        # print(pred_y)

        pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))

        # print(y)
        # print(pred_y)
        correct += np.sum(y == pred_y)

    # print(correct, dataset_size)
    acc = 100.0 * (float(correct) / float(dataset_size))
    # total_l = total_loss / dataset_size

    return acc

class Server(object):

    def __init__(self, conf, eval_dataset, w_size):
        self.public_key = public_key
        self.private_key = private_key

        self.conf = conf

        self.global_model = models.LR_Model(public_key=public_key, w_size=w_size)

        self.eval_x = eval_dataset[0]

        self.eval_y = eval_dataset[1]

    def get_public_key(self):
        return self.public_key

    def model_aggregate(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["lambda"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer
    #     self.global_model.weights = models.decrypt_vector(private_key, self.global_model.encrypt_weights)

    # def model_eval(self):
    #
    #     total_loss = 0.0
    #     correct = 0
    #     dataset_size = 0
    #
    #     batch_num = int(self.eval_x.shape[0] / self.conf["batch_size"])
    #
    #     self.global_model.weights = models.decrypt_vector(private_key, self.global_model.encrypt_weights)
    #     print(self.global_model.weights)
    #
    #     for batch_id in range(batch_num):
    #         x = self.eval_x[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]]
    #         x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    #         y = self.eval_y[batch_id * self.conf["batch_size"]: (batch_id + 1) * self.conf["batch_size"]].reshape(
    #             (-1, 1))
    #
    #         dataset_size += x.shape[0]
    #
    #         wxs = x.dot(self.global_model.weights)
    #
    #         pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]
    #
    #         # print(pred_y)
    #
    #         pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))
    #
    #         # print(y)
    #         # print(pred_y)
    #         correct += np.sum(y == pred_y)
    #
    #     # print(correct, dataset_size)
    #     acc = 100.0 * (float(correct) / float(dataset_size))
    #     # total_l = total_loss / dataset_size
    #
    #     return acc


    @staticmethod
    def re_encrypt(w):
        return models.encrypt_vector(public_key, models.decrypt_vector(private_key, w))