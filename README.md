# DEEP RESIDUAL ECHO SUPPRESSION WITH A TUNABLE TRADEOFF BETWEEN SIGNAL DISTORTION AND ECHO SUPPRESSION (Accepted to ICASSP 2021 Conference)
### Amir Ivry, Prof. Israel Cohen, Dr. Baruch Berdugo <br/> 
#### Andrew and Erna Viterbi Faculty of Electrical and Computer Engineering, Technion - Israel Institute of Technology
> This research proposes a residual echo suppression method using a UNet neural network that directly maps the outputs of a linear acoustic echo canceler to the desired signal in the spectral domain. This system embeds a design parameter that allows a tunable tradeoff between the desired-signal distortion and residual echo suppression in double-talk scenarios. The system employs 136 thousand parameters, and requires 1.6 Giga floating-point operations per second and 10 Mega-bytes of memory. The implementation satisfies both the timing requirements of the AEC challenge and the computational and memory limitations of on-device applications. Experiments are conducted with 161 h of data from the AEC challenge database and from real independent recordings. We demonstrate the performance of the proposed system in real-life conditions and compare it with two competing methods regarding echo suppression and desired-signal distortion, generalization to various environments, and robustness to high echo levels <br/> We share the code here for reproducability and it is our hope you will also find it instructive for speech residual echo suppression. You are also encouraged to refer to the more elaborated published paper on [Arxiv](https://arxiv.org/pdf/2106.13531.pdf).
> Demo can be found [_here_](https://soundcloud.com/ai4audio/sets/objective-metrics-to-evaluate-residual-echo-suppression-during-double-talk). 


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)


## General Information
This code implements a deep learning-based residual echo suppressor that is meant to preserve desired speech and cancel echo in mono acoustic echo cancellation setups. This implementation is computationaly lean, and embeds a training objective function with a dedicated design parameter. This parameter dynamically controls the trade-off between speech distortion and echo suppression that the system exhibits. A trained pytorch model is supplied, with scripts that enable training from scratch, fine-tuning, and inference.


## Technologies Used
- Python 3.8.6
- scipy==1.7.0
- matplotlib==3.4.2
- numpy==1.21.0


## Setup
To prepare for usage, the user should follow these steps:
- Clone this repo
- Set up a virtual environment and run: `pip install -r requirements.txt`
- Create a parent direcotry and assign its relative path to `data_path` variable inside `main.py`
- Inside this dir, locate each example in a separate subdirectory with a unique name. Every subfolder must contain the near end speech, the RES input before the system gain, and the RES prediction after the system gain, all in the time domain (.wav format). The names of these 3 files are user-dependent, but should not vary accross subdirectories. Assign the files names to `patterns` inside `main.py` and ensure the order of appearance is as given above.

The last two steps are explained with the following demo example. After cloning the repo and setting up the venv, a parent directory called _Demo_ is created and its name is assigned to `data_path`. Inside _Demo_, there are 5 subdirectories uniquely named _Example 1_ through _Exmaple 5_. Inside _Example 1_, for instance, there are several files, 3 of which are essential - _near_end_speech.wav_, _res_input.wav_, and _res_prediction.wav_. These names are consistent across all subdirectores and are assigned in *that order* to `patterns`.


| ![image](https://user-images.githubusercontent.com/22732198/125336393-64a29000-e356-11eb-910d-1b7af4520549.png) |
|:--:|
| *Snippet of a demo project setup, prepared for usage.* |


## Usage
After setup, the user should follow these steps to use the code:
- run: `main.py`
- The log file `evaluation_metrics.txt` (the name is hard-coded) will appear inside the path assigned to the variable `data_path` (see _Setup_ for details). It contains the mean and standard deviation values of the DSML and RESL measures for every subdirectory.


| ![image](https://user-images.githubusercontent.com/22732198/125337140-4ab57d00-e357-11eb-91d7-40c16f2864f8.png) |
|:--:|
| *Snippet of the produced `evaluation_metrics.txt` file after demo run.* |


## Project Status
Project is complete. Occassional fine-tuning may appear (see _Room for Improvement_).


## Room for Improvement
Future release may include:
- Automatic double-talk detector to filter out irrelavant single-talk segments 
- Enhance user experience by adding more features and GUI
- Permit running the code from shell


## Acknowledgements
This research was supported by the Pazy Research Foundation, the Israel Science Foundation (ISF), and the International Speech Communication Association (ISCA). We would also like to thank stem audio for their technical support.<br/> If you use this repo or other instance of this research, please cite the following: <br/>
`@inproceedings{ivry2021objective,`<br/>
  `title={Objective Metrics to Evaluate Residual-Echo Suppression During Double-Talk},`<br/>
  `author={Ivry, Amir and Cohen, Israel and Berdugo, Baruch},`<br/>
  `booktitle={WASPAA},`<br/>
  `year={2021},`<br/>
  `organization={IEEE}`<br/>
`}`


## Contact
Created by [Amir Ivry](https://www.linkedin.com/in/amirivry/) - feel free to contact me also via [amirivry@gmail.com](amirivry@gmail.com).
