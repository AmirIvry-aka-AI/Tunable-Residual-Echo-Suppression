function [] = saveFeaturesList(input, config, varargin)

%% Doc:
% This function saves the network-prepared structures to the folder
% specified by config in a uni-naming approach. 
% INPUTS:
% input - input features prepared for network
% config - system configuration
% varargin - (target), target features prepared for network, given training mode

    featFolder      = config.featFolder;
    save([featFolder,'input.mat'], 'input', '-v7.3');
    
    if ~isempty(varargin)
        target      = varargin{1};
        save([featFolder,'target.mat'], 'target', '-v7.3');
    end