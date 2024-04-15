#权限验证模块
import torch


class Guest:
    def __init__(self):
        pass

    def combine_params(self, models):
        fc1_sum = torch.zeros_like(models[0][0][1].data)
        fc2_sum = torch.zeros_like(models[0][2][1].data)

        for model in models:
            fc1_sum += model[0][1].data
            fc2_sum += model[2][1].data

        com_para_fc1 = fc1_sum / len(models)
        com_para_fc2 = fc2_sum / len(models)

        return com_para_fc1, com_para_fc2

    # def combine_params(self, models):


        # para_A = train_model_parameter[0]
        # fc1_wA = models[0][0][1].data
        # fc1_wB = models[1][0][1].data
        # fc1_wC = models[2][0][1].data
        # #
        # fc2_wA = models[0][2][1].data
        # fc2_wB = models[1][2][1].data
        # fc2_wC = models[2][2][1].data
        # #
        # com_para_fc1 = (fc1_wA + fc1_wB + fc1_wC) / 3
        # com_para_fc2 = (fc2_wA + fc2_wB + fc2_wC) / 3
        # return com_para_fc1, com_para_fc2

