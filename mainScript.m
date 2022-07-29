%% Documentation:
% This script applies training and testing of the RES system described in:
% "Deep residual echo suppression with a tunable tradeoff between signal
% distortion and echo suppression", ICASSP, 2021.

% USER INPUTS:
% dataFolder - name of a user-created directory (on script-level), in which two
% .pcm files must be present named 'mic.pcm', 'ref.pcm'. If mode below is
% 'train', then it also must contain 'target.pcm'
% mode - 'train'/'test'

%% Initialize
clear; close all; clc;
addpath(genpath(pwd));    
warning off;

%% User inputs
dataFolder              = 'Full DB\TIMIT TEST -10\'; 
mode                    = 'test'; %'train'

%% Configuration
config                  = configSystem(dataFolder, mode, 'unet');

%% Create database (.pcm, .wav, STFT, features) - this process should be applied only once
createDatabase(dataFolder, config);

%% Run Python

%% Synthesis - at this stage the network's predictions are available
if(strcmp(mode,'test'))
    pause;
    disp('Please run python code before you continue.');
    synthesizePredictions(config);
end