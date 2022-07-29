import torch
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from utils import load_test_data, create_test_loader
from classes.unet_model import UNet
import os

parent_dir = '..'
config = sio.loadmat(os.path.join(parent_dir, 'config.mat'))['config']
model_folder = os.path.join(parent_dir, config['modelFolder'][0][0][0])
predictions_folder = os.path.join(parent_dir, config['netFolder'][0][0][0], 'predictions')
feat_folder = config['featFolder'][0][0][0]
architecture = config['architecture'][0][0][0]
mode = config['mode'][0][0][0]
melFeatures = config['melFeatures'][0][0][0][0]
input_path = os.path.join(parent_dir, feat_folder, 'input.mat')
target_path = os.path.join(parent_dir, feat_folder, 'target.mat')
if architecture == 'fcnn':
    batch_size = 16
else:
    batch_size = 4
lr = 5e-4
alpha = 1

test_input, test_target = load_test_data(input_path)
test_loader = create_test_loader(test_input, test_target, architecture, mode, batch_size_=batch_size)

if architecture == 'fcnn':
    from classes.fcnn import Fcnn
    net = Fcnn()
    prediction = np.empty((test_target.size()[1], melFeatures))
else:
    from classes.unet_model import UNet
    net = UNet(2, 1)
    prediction = np.empty((test_target.size()[2]-29, test_target.size()[1]))

net.load_state_dict(torch.load(model_folder+r'\model.pt'))  #, map_location=torch.device('cpu')))
net.eval()
net = net.cuda()

counter = 0
for data, target in tqdm(test_loader):
    data = data.cuda()
    tmp_output = net(data).cpu().detach().numpy()
    if architecture == 'fcnn':
        prediction[counter:counter + np.shape(tmp_output)[0], :] = tmp_output
    else:
        prediction[counter:counter + np.shape(tmp_output)[0], :] = tmp_output[:, 0, :, -1]
    counter += np.shape(tmp_output)[0]
np.save(predictions_folder + '/prediction', prediction)
