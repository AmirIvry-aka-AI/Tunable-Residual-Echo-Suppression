from __future__ import absolute_import
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import h5py


def load_train_data(input_path, target_path):
    # Loading the train data

    # numExamples = 8285
    # test_data_db = torch.from_numpy(np.empty((2, 161, 9820000))).float()
    # test_tag_db = torch.from_numpy(np.empty((1, 161, 9820000))).float()
    # test_data_fname = inputPath + '/input_' + str(0) + '.mat'
    f = h5py.File(input_path)
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v)
    train_input_db = torch.from_numpy(np.transpose(arrays['input'])).float()

    f = h5py.File(target_path)
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v)
    train_target_db = torch.from_numpy(np.transpose(arrays['target'])).float()
    # temp = np.transpose(temp)
    # temp = torch.from_numpy(temp).float()
    # test_data_db[:, :, 0:temp.size()[2]] = temp

    # test_tag_fname = target_path + '/target_' + str(0) + '.mat'
    # f = h5py.File(test_tag_fname)
    # arrays = {}
    # for k, v in f.items():
    #     arrays[k] = np.array(v)
    # test_tag_temp = arrays['targetAmpPrepared']
    # test_tag_temp = np.moveaxis(test_tag_temp, 1, 0)
    # test_tag_temp = np.expand_dims(test_tag_temp, axis=0)
    # test_tag_temp = torch.from_numpy(test_tag_temp).float()
    # # print(train_tag_temp.size())
    # # print(train_tag_db.size())
    # test_tag_db[:, :, 0:test_tag_temp.size()[2]] = test_tag_temp
    #
    # counter = test_tag_temp.size()[2]
    #
    # print("Train data and tags:")
    # for i in range(1, 0 + (numExamples - 1) + 1):
    #     print(100 * (i - 0) / ((numExamples - 1) + 1))
    #     test_data_fname = input_path + '/input_' + str(i) + '.mat'
    #     f = h5py.File(test_data_fname)
    #     arrays = {}
    #     for k, v in f.items():
    #         arrays[k] = np.array(v)
    #     temp = arrays['inputAmpPrepared']
    #     temp = np.transpose(temp)
    #     temp = torch.from_numpy(temp).float()
    #     tempSize = temp.size()[2]
    #     #        print(tempSize)
    #     #        print(tempSize+dataSize)
    #     #        print(train_data_temp.size())
    #     #        train_data_db.append(train_data_temp, axis=2)
    #     test_data_db[:, :, counter:counter + tempSize] = temp
    #     #        torch.cat((train_data_db, torch.from_numpy(train_data_temp).float()), dim=2)
    #
    #     test_tag_fname = target_path + '/target_' + str(i) + '.mat'
    #     f = h5py.File(test_tag_fname)
    #     arrays = {}
    #     for k, v in f.items():
    #         arrays[k] = np.array(v)
    #     test_tag_temp = arrays['targetAmpPrepared']
    #     test_tag_temp = np.moveaxis(test_tag_temp, 1, 0)
    #     test_tag_temp = np.expand_dims(test_tag_temp, axis=0)
    #     test_tag_temp = torch.from_numpy(test_tag_temp).float()
    #     tempSize = test_tag_temp.size()[2]
    #     #        train_tag_db.append(train_tag_temp, axis=2)
    #     test_tag_db[:, :, counter:counter + tempSize] = test_tag_temp
    #     #        torch.cat((train_tag_db, torch.from_numpy(train_tag_temp).float()), dim=2)
    #
    #     counter = counter + tempSize
    #     # print(counter)
    #
    # test_data_db = test_data_db[:, :, 0:counter]
    # test_tag_db = test_tag_db[:, :, 0:counter]

    print(train_input_db.size())
    print(train_target_db.size())
    # print(counter)

    return train_input_db, train_target_db


def load_test_data(input_path):
    # Loading the train data

    # numExamples = 8285
    # test_data_db = torch.from_numpy(np.empty((2, 161, 9820000))).float()
    # test_tag_db = torch.from_numpy(np.empty((1, 161, 9820000))).float()
    # test_data_fname = inputPath + '/input_' + str(0) + '.mat'
    f = h5py.File(input_path)
    arrays = {}
    for k, v in f.items():
        arrays[k] = np.array(v)
    test_input_db = torch.from_numpy(np.transpose(arrays['input'])).float()

    # f = h5py.File(target_path)
    # arrays = {}
    # for k, v in f.items():
    #     arrays[k] = np.array(v)
    # test_target_db = torch.from_numpy(np.transpose(arrays['target'])).float()
    test_target_db = torch.from_numpy(np.empty((test_input_db.size()))).float()

    print(test_input_db.size())
    print(test_target_db.size())

    return test_input_db, test_target_db


#    train_data_db = torch.from_numpy(np.empty((2,161,1000*5))).float()
#    train_tag_db = torch.from_numpy(np.empty((1,161,1000*5))).float()
#
#    train_data_fname = inputPath+'/input_'+str(500)+'.mat'
#    f = h5py.File(train_data_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    train_data_temp = arrays['inputAmpPrepared']
#    train_data_temp = np.transpose(train_data_temp)
#    train_data_temp = torch.from_numpy(train_data_temp).float()
#    train_data_db[:,:,0:train_data_temp.size()[2]] = train_data_temp
#    
#    train_tag_fname = targetPath+'/target_'+str(500)+'.mat'
#    f = h5py.File(train_tag_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    train_tag_temp = arrays['targetAmpPrepared']
#    train_tag_temp = np.moveaxis(train_tag_temp, 1, 0)
#    train_tag_temp = np.expand_dims(train_tag_temp, axis=0)
#    train_tag_temp = torch.from_numpy(train_tag_temp).float()
#    #print(train_tag_temp.size())
#    #print(train_tag_db.size())
#    train_tag_db[:,:,0:train_tag_temp.size()[2]] = train_tag_temp
#    
#    counter = train_tag_temp.size()[2]
#    
#    print("Train data and tags:")
#    for i in range(501,505):#9999+1):
#        print(100*(i-500)/(9999+1-500))
#        train_data_fname = inputPath+'/input_'+str(i)+'.mat'
#        f = h5py.File(train_data_fname)
#        arrays = {}
#        for k, v in f.items():
#            arrays[k] = np.array(v)
#        train_data_temp = arrays['inputAmpPrepared']
#        train_data_temp = np.transpose(train_data_temp)
#        train_data_temp = torch.from_numpy(train_data_temp).float()
#        tempSize = train_data_temp.size()[2]
##        print(tempSize)
##        print(tempSize+dataSize)
##        print(train_data_temp.size())
##        train_data_db.append(train_data_temp, axis=2)
#        train_data_db[:,:, counter:counter+tempSize] = train_data_temp
##        torch.cat((train_data_db, torch.from_numpy(train_data_temp).float()), dim=2)
#    
#        train_tag_fname = targetPath+'/target_'+str(i)+'.mat'
#        f = h5py.File(train_tag_fname)
#        arrays = {}
#        for k, v in f.items():
#            arrays[k] = np.array(v)
#        train_tag_temp = arrays['targetAmpPrepared']
#        train_tag_temp = np.moveaxis(train_tag_temp, 1, 0)
#        train_tag_temp = np.expand_dims(train_tag_temp, axis=0)
#        train_tag_temp = torch.from_numpy(train_tag_temp).float()
#        tempSize = train_tag_temp.size()[2]
##        train_tag_db.append(train_tag_temp, axis=2)
#        train_tag_db[:,:, counter:counter+tempSize] = train_tag_temp
##        torch.cat((train_tag_db, torch.from_numpy(train_tag_temp).float()), dim=2)
#        
#        counter = counter + tempSize
#        #print(counter)
#    
#    train_data_db = train_data_db[:,:,0:counter]
#    train_tag_db = train_tag_db[:,:,0:counter]
#
#    print(train_data_db.size())
#    print(train_tag_db.size())
#    print(counter)
#    
#    return train_data_db, train_tag_db
#

# def load_test_data_OneByOne(inputPath, targetPath):
#     # Loading the test data
#     numExamples = 30
#     test_data_db = torch.from_numpy(np.empty((2, 161, 600000))).float()
#     #    test_tag_db = torch.from_numpy(np.empty((1,161,2000*numExamples))).float()
#
#     test_data_fname = inputPath + '/input_' + str(0) + '.mat'
#     f = h5py.File(test_data_fname)
#     arrays = {}
#     for k, v in f.items():
#         arrays[k] = np.array(v)
#     test_data_temp = arrays['inputAmpPrepared']
#     test_data_temp = np.transpose(test_data_temp)
#     test_data_temp = torch.from_numpy(test_data_temp).float()
#     test_data_db[:, :, 0:test_data_temp.size()[2]] = test_data_temp
#
#     #    test_tag_fname = targetPath+'/target_'+str(0)+'.mat'
#     #    f = h5py.File(test_tag_fname)
#     #    arrays = {}
#     #    for k, v in f.items():
#     #        arrays[k] = np.array(v)
#     #    test_tag_temp = arrays['targetAmpPrepared']
#     #    test_tag_temp = np.moveaxis(test_tag_temp, 1, 0)
#     #    test_tag_temp = np.expand_dims(test_tag_temp, axis=0)
#     #    test_tag_temp = torch.from_numpy(test_tag_temp).float()
#     #    #print(train_tag_temp.size())
#     #    #print(train_tag_db.size())
#     #    test_tag_db[:,:,0:test_tag_temp.size()[2]] = test_tag_temp
#
#     counter = test_data_temp.size()[2]
#
#     print("Train data and tags:")
#     for i in range(1, (numExamples - 1) + 1):
#         print(100 * (i) / ((numExamples - 1) + 1))
#         test_data_fname = inputPath + '/input_' + str(i) + '.mat'
#         f = h5py.File(test_data_fname)
#         arrays = {}
#         for k, v in f.items():
#             arrays[k] = np.array(v)
#         test_data_temp = arrays['inputAmpPrepared']
#         test_data_temp = np.transpose(test_data_temp)
#         test_data_temp = torch.from_numpy(test_data_temp).float()
#         tempSize = test_data_temp.size()[2]
#         #        print(tempSize)
#         #        print(tempSize+dataSize)
#         #        print(train_data_temp.size())
#         #        train_data_db.append(train_data_temp, axis=2)
#         test_data_db[:, :, counter:counter + tempSize] = test_data_temp
#         #        torch.cat((train_data_db, torch.from_numpy(train_data_temp).float()), dim=2)
#
#         #        test_tag_fname = targetPath+'/target_'+str(i)+'.mat'
#         #        f = h5py.File(test_tag_fname)
#         #        arrays = {}
#         #        for k, v in f.items():
#         #            arrays[k] = np.array(v)
#         #        test_tag_temp = arrays['targetAmpPrepared']
#         #        test_tag_temp = np.moveaxis(test_tag_temp, 1, 0)
#         #        test_tag_temp = np.expand_dims(test_tag_temp, axis=0)
#         #        test_tag_temp = torch.from_numpy(test_tag_temp).float()
#         #        tempSize = test_tag_temp.size()[2]
#         ##        train_tag_db.append(train_tag_temp, axis=2)
#         #        test_tag_db[:,:, counter:counter+tempSize] = test_tag_temp
#         ##        torch.cat((train_tag_db, torch.from_numpy(train_tag_temp).float()), dim=2)
#
#         counter = counter + tempSize
#         # print(counter)
#
#     test_data_db = test_data_db[:, :, 0:counter]
#     #    test_tag_db = test_tag_db[:,:,0:counter]
#
#     test_tag_db = torch.from_numpy(np.empty((test_data_db.size()))).float()
#     print(test_data_db.size())
#     print(test_tag_db.size())
#     print(counter)
#
#     return test_data_db, test_tag_db


# def load_train_data(path, inputSize, targetSize):
#    # Loading the train data
#    print("Train data and tags:")
#    train_data_fname = pjoin(path, 'input.mat')
#    f = h5py.File(train_data_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    train_data_db = arrays['inputAmpPrepared']
#    train_data_db = np.transpose(train_data_db)
#    train_data_db = torch.from_numpy(train_data_db).float()
#    print(train_data_db.size())
#
#    train_tag_fname = pjoin(path, 'target.mat')
#    f = h5py.File(train_tag_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    train_tag_db = arrays['targetAmpPrepared']
#    train_tag_db = np.moveaxis(train_tag_db, 1, 0)
#    train_tag_db = np.expand_dims(train_tag_db, axis=0)
#    train_tag_db = torch.from_numpy(train_tag_db).float()
#    print(train_tag_db.size())
#
#    return train_data_db, train_tag_db
#
# def load_test_data(path, inputSize, targetSize):
#    # Loading the test data
#    print("Test data and tags:")
#    test_data_fname = pjoin(path, 'input.mat')
#    f = h5py.File(test_data_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    test_data_db = arrays['inputAmpPrepared']
#    test_data_db = np.transpose(test_data_db)
#    test_data_db = torch.from_numpy(test_data_db).float()
#    print(test_data_db.size())
#    
#    test_tag_fname = pjoin(path, 'target.mat')
#    f = h5py.File(test_tag_fname)
#    arrays = {}
#    for k, v in f.items():
#        arrays[k] = np.array(v)
#    test_tag_db = arrays['targetAmpPrepared']
#    test_tag_db = np.moveaxis(test_tag_db, 1, 0)
#    test_tag_db = np.expand_dims(test_tag_db, axis=0)
#    test_tag_db = torch.from_numpy(test_tag_db).float()
#    print(test_tag_db.size())
#    
#    return test_data_db, test_tag_db

def create_train_valid_loader(input_data, target_data, architecture, batch_size_, valid_size_, mode):
    # Creating out dataset
    from classes.CreateDataset import CreateDataset
    train_data = CreateDataset(input_data, target_data, architecture, mode, transform=None)

    # Set Batch Size
    batch_size = batch_size_

    # Percentage of training set to use as validation
    valid_size = valid_size_

    # obtain training indices that will be used for validation
    if architecture == 'fcnn':
        num_train = input_data.size()[1]
    else:
        num_train = input_data.size()[2] // 30

    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    print(num_train)
    print(split)

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def create_test_loader(input_data, target_data, architecture, mode, batch_size_):
    # Creating out dataset
    from classes.CreateDataset import CreateDataset
    test_data = CreateDataset(input_data, target_data, architecture, mode, transform=None)

    # Set Batch Size
    batch_size = batch_size_

    # prepare data loader
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader


def training_network(lr_, n_epochs, train_loader, valid_loader, model_name, alpha, arch):
    # creating the U-net model
    # from keras import backend as k

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('\nCUDA is not available.  Training on CPU ...')
    else:
        print('\nCUDA is available!  Training on GPU ...')

    # Either UNet or FCNN
    if arch == 'fcnn':
        from classes.fcnn import Fcnn
        net = Fcnn()
    else:
        from classes.unet_model import UNet
        net = UNet(2, 1)

    if train_on_gpu:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_)

    # criterion = nn.MSELoss()

    def costum_loss(pred, gr_truth):
        return torch.mean(torch.square(pred - gr_truth) +
                          alpha * torch.square(pred) + (alpha > 0) * 0.1 * torch.square(pred - torch.mean(pred)))
    criterion = costum_loss

    # def custom_loss(y_pred, y_true):
    #     return k.mean(k.square(y_pred - y_true) +
    #                   alpha*k.square(y_pred) + (alpha > 0)*0.1*k.square(y_pred - k.mean(y_pred)))
    # criterion = custom_loss

    # print('current lr = {:.0e}'.format(lr_))

    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    best_epoch = 1
    for epoch in range(1, n_epochs + 1):
        print('Epoch: ' + str(epoch) + '...')
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        net.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            #            print(data.size())
            #            print(target.size())
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the batch loss
            # print(output.size())
            # print(target.size())
            loss = criterion(output,
                             target)  # + np.square(output).cpu() + 0.1*np.square(output - np.mean(output)).cpu()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            print('Epoch #' + str(epoch) + ' updated training loss = ' + str(loss.item()))
            # print(data.shape)
            # print(output.shape)
            # print(target.shape)
        ######################
        # validate the model #
        ######################
        net.eval()
        # a = print(net)
        # np.save(dataPath + '/net_weigths.txt', net)

        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            # calculate the batch loss
            loss = criterion(output,
                             target)  # + np.square(output).cpu() + 0.1 * np.square(output - np.mean(output)).cpu()
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

            print('Epoch #' + str(epoch) + ' updated validation loss = ' + str(loss.item()))

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        # torch.save(net.state_dict(), model_name+'_Epoch_'+str(epoch)+'.pt')
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            # savePTH(net)
            torch.save(net.state_dict(), model_name)
            valid_loss_min = valid_loss
            best_epoch = epoch

    return train_losses, valid_losses, best_epoch


def transfer_learning(model_path, lr_, n_epochs, train_loader, valid_loader, model_name):
    # creating the U-net model
    from classes.unet_model import UNet
    # import tensorflow as tf
    # net = UNet(1, 2)
    net = UNet(2, 1)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    # training on GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('\nCUDA is not available.  Training on CPU ...')
    else:
        print('\nCUDA is available!  Training on GPU ...')

    if train_on_gpu:
        net.cuda()

    # Loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr_)
    print('current lr = {:.0e}'.format(lr_))

    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch: ' + str(epoch) + '...')
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        net.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            n = torch.numel(output)
            # print(np.linalg.norm(np.square(output.detach().cpu().numpy())))
            # calculate the batch loss
            loss = criterion(output, target) + (1 / n) * np.square(np.linalg.norm(output.detach().cpu().numpy()))
            # loss = torch.from_numpy(loss).cuda()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            print('Epoch #' + str(epoch) + ' updated training loss = ' + str(loss.item()))
            # print(data.shape)
            # print(output.shape)
            # print(target.shape)
        ######################
        # validate the model #
        ######################
        net.eval()
        # a = print(net)
        # np.save(dataPath + '/net_weigths.txt', net)

        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)
            n = torch.numel(output)
            # calculate the batch loss
            loss = criterion(output, target) + (1 / n) * np.square(np.linalg.norm(output.detach().cpu().numpy()))
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

            print('Epoch #' + str(epoch) + ' updated validation loss = ' + str(loss.item()))

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        # torch.save(net.state_dict(), model_name+'_Epoch_'+str(epoch)+'.pt')
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            # savePTH(net)
            torch.save(net.state_dict(), model_name)
            valid_loss_min = valid_loss
            bestEpoch = epoch
            print('Best Epoch = ' + str(bestEpoch))

    return train_losses, valid_losses, bestEpoch
