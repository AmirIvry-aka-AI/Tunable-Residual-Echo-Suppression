from classes.unet_model import UNet
from torchsummary import summary
import torch 
from ptflops import get_model_complexity_info

net = UNet(2,1)
inputSize = (2, 161, 30)
summary(net, inputSize)

with torch.cuda.device(0):
  net = UNet(2,1)
  flops, params = get_model_complexity_info(net, inputSize, as_strings=True, print_per_layer_stat=False, verbose=True)
  print('{:<30}  {:<8}'.format('FLOPs: ', flops))

from denoiser.demucs import Demucs

import torchsummary
dns = Demucs().cuda()
torchsummary.summary(dns, (1,128))

import numpy as np
batchSize = 4
device = torch.device("cpu")
net.to(device)
dummy_input = torch.randn(batchSize, 2, 161, 30, dtype=torch.float).to(device)
# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#starter, ender = torch.Event(enable_timing=True), torch.Event(enable_timing=True)

repetitions = 100
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = net(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = net(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / (batchSize*repetitions)
std_syn = np.std(timings)
print(mean_syn)

import time
s = time.time()
_ = net(dummy_input)
curr = (time.time()-s)*1000
print(curr/32)

net = UNet(1,1)
inputSize = (1, 161, 30)
summary(net, inputSize)