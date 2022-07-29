# DEEP RESIDUAL ECHO SUPPRESSION WITH A TUNABLE TRADEOFF BETWEEN SIGNAL DISTORTION AND ECHO SUPPRESSION (Accepted to ICASSP 2021 Conference)
### Amir Ivry, Prof. Israel Cohen, Dr. Baruch Berdugo <br/> 
#### Andrew and Erna Viterbi Faculty of Electrical and Computer Engineering, Technion - Israel Institute of Technology
> This research proposes a residual echo suppression method using a UNet neural network that directly maps the outputs of a linear acoustic echo canceler to the desired signal in the spectral domain. This system embeds a design parameter that allows a tunable tradeoff between the desired-signal distortion and residual echo suppression in double-talk scenarios. The system employs 136 thousand parameters, and requires 1.6 Giga floating-point operations per second and 10 Mega-bytes of memory. The implementation satisfies both the timing requirements of the AEC challenge and the computational and memory limitations of on-device applications. Experiments are conducted with 161 h of data from the AEC challenge database and from real independent recordings. We demonstrate the performance of the proposed system in real-life conditions and compare it with two competing methods regarding echo suppression and desired-signal distortion, generalization to various environments, and robustness to high echo levels <br/> We share the code here for reproducability and it is our hope you will also find it instructive for speech residual echo suppression. You are also encouraged to refer to the more elaborated published paper on [Arxiv](https://arxiv.org/pdf/2106.13531.pdf).
> Demo can be found [_here_](https://soundcloud.com/ai4audio/sets/objective-metrics-to-evaluate-residual-echo-suppression-during-double-talk). 


## Table of Contents
* [General Info](#general-information)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)


## General Information
This code implements a deep learning-based residual echo suppressor that is meant to preserve desired speech and cancel echo in mono acoustic echo cancellation setups. This implementation is computationaly lean, and embeds a training objective function with a dedicated design parameter. This parameter dynamically controls the trade-off between speech distortion and echo suppression that the system exhibits. A pytorch model is provided with a Python-MATLAB API that allows training and inference.


## Setup
To prepare for usage, the user should follow these steps:
- Clone this repo
- Create a MATLAB project with the following folder leveling, where 'data_folder' contains two subfolders - 'train' and 'test':
| ![folder_structure](https://user-images.githubusercontent.com/22732198/181769787-93920292-1477-451d-b35f-5655dca0a4ed.PNG) |
|:--:|
| *MATLAB project leveling.* |
- The 'train' folder holds the 'mic.pcm', 'ref.pcm', and 'target.pcm' files. The 'test' folder holds the same without the 'target.pcm'
- Set up a virtual environment and run: `pip install -r requirements.txt`
- Follow MATLAB's 'main_script.m' documentation

## Usage
Open mainscript.m and follow internal MATLAB's documentation on how to insert user parameters and how to employ the PYTHON API. The user will be required to mention the desired scenario (training/fine-tuning/testing) and provide path to data.

| ![image](https://user-images.githubusercontent.com/22732198/125337140-4ab57d00-e357-11eb-91d7-40c16f2864f8.png) |
|:--:|
| *Snippet of the produced `evaluation_metrics.txt` file after demo run.* |


## Project Status
Project is complete. Occassional updating of models trained on more data may occur.


## Acknowledgements
This research was supported by the Pazy Research Foundation, the Israel Science Foundation (ISF), and the International Speech Communication Association (ISCA). We would also like to thank stem audio for their technical support.<br/> If you use this repo or other instance of this research, please cite the following: <br/>
`@inproceedings{ivry2021objective,`<br/>
  `title={DEEP RESIDUAL ECHO SUPPRESSION WITH A TUNABLE TRADEOFF BETWEEN SIGNAL DISTORTION AND ECHO SUPPRESSION},`<br/>
  `author={Ivry, Amir and Cohen, Israel and Berdugo, Baruch},`<br/>
  `booktitle={ICASSP},`<br/>
  `year={2021},`<br/>
  `organization={IEEE}`<br/>
`}`


## Contact
Created by [Amir Ivry](https://www.linkedin.com/in/amirivry/) - feel free to contact me also via [amirivry@gmail.com](amirivry@gmail.com).
