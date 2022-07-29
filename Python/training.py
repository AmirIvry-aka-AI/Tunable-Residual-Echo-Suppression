from utils import load_train_data, create_train_valid_loader, training_network
import scipy.io as sio
import os
import numpy as np
# import matplotlib.pyplot as plt

# Import config.mat from MATLAB
parent_dir = '..'
config = sio.loadmat(os.path.join(parent_dir, 'config.mat'))['config']
model_folder = os.path.join(parent_dir, config['netFolder'][0][0][0], 'model')
feat_folder = config['featFolder'][0][0][0]
architecture = config['architecture'][0][0][0]
mode = config['mode'][0][0][0]
input_path = os.path.join(parent_dir, feat_folder, 'input.mat')
target_path = os.path.join(parent_dir, feat_folder, 'target.mat')
if architecture == 'fcnn':
    batch_size = 16
else:
    batch_size = 4
lr = 5e-4
alpha = 0

# Prepare database for network
train_input, train_target = load_train_data(input_path, target_path)
train_loader, valid_loader = \
    create_train_valid_loader(train_input, train_target, architecture, batch_size_=batch_size, valid_size_=0.2, mode=mode)

# Train network and save outcome
train_losses, valid_losses, best_epoch = \
    training_network(lr_=lr, n_epochs=20, train_loader=train_loader, valid_loader=valid_loader,
                     model_name=model_folder + '/model.pt', alpha=alpha, arch=architecture)

np.save(model_folder + '/train_loss', train_losses)
np.save(model_folder + '/valid_loss', valid_losses)
np.save(model_folder + '/best_epoch', best_epoch)
#
# plt.figure()
# fig = plt.gcf()
# plt.plot(train_losses, label='Training loss')
# plt.plot(valid_losses, label='Validation loss')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(frameon=False)
# plt.title('Loss with lr = {:.0e}'.format(lr))
# plt.show()
# fig.savefig(model_folder + '/performance.png')