from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from sklearn import preprocessing
import torch


# Inherit Dataset class for efficient data generation
# class CreateDatasetTrain(Dataset):
class CreateDataset(Dataset):

    def __init__(self, input_data, target_data, architecture, mode, transform=None):
        super().__init__()
        self.input_data = input_data
        self.target_data = target_data
        self.transform = transform
        self.architecture = architecture
        self.mode = mode

    def __len__(self):
        if self.architecture == 'fcnn':
            return self.input_data.size()[1]
        else:
            if self.mode == 'train':
                return self.input_data.size()[2] // 30
            else:
                return self.input_data.size()[2] - 29

    def __getitem__(self, index):
        if self.architecture == 'fcnn':
            batch_input = self.input_data[:, index]
            batch_target = self.target_data[:, index]
        else:
            if self.mode == 'train':
                batch_input = self.input_data[:, :, (30 * index):(30 * (index + 1))]
                batch_target = self.target_data[:, :, (30 * index):(30 * (index + 1))]
            else:
                batch_input = self.input_data[:, :, index:index + 30]
                batch_target = self.target_data[:, :, index:index + 30]
        # batch_data = self.df_data[:,:,index:index+30]
        # batch_tag = self.df_tag[:,:,index:index+30]
        # if self.transform is not None:
        # image = self.transform(image)
        return batch_input, batch_target


# class CreateDatasetTest(Dataset):
#     def __init__(self, input_data, target_data, architecture, transform=None):
#         super().__init__()
#         self.input_data = input_data
#         self.target_data = target_data
#         self.transform = transform
#         self.architecture = architecture
#
#     def __len__(self):
#         if self.architecture == 'fcnn':
#             return self.input_data.size()[1]
#         else:
#             return self.input_data.size()[2] // 30
#
#     def __getitem__(self, index):
#         if self.architecture == 'fcnn':
#             batch_input = self.input_data[:, index]
#             batch_target = self.target_data[:, index]
#         else:
#             batch_input = self.input_data[:, :, (30 * index):(30 * (index + 1))]
#             batch_target = self.target_data[:, :, (30 * index):(30 * (index + 1))]
#
#         return batch_input, batch_target
