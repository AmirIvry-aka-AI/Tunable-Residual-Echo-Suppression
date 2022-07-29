function config = configSystem(dataFolder, varargin)

%% DOC: 
% This function creates system configuration such as folders and parameters.
% INPUTS:
% dataFolder - the folder in which the .pcm files are located in
% varargin = (mode, architecture):
% mode - 'train' (def.) /'test'
% architecture - 'fcnn' (def.) /'unet'
% OUTPUT:
% System configuration - a struct with relevant fields for system operation

    mode            = 'train'; 
    architecture    = 'fcnn';
    if nargin       == 3
        mode        = lower(varargin{1}); 
        architecture= lower(varargin{2});
    end
    if nargin       == 2
        mode        = lower(varargin{1});
    end
    
    config.generatePCM  = 1;
    config.generateWAV  = 1;
    config.generateFeat = 1;
    config.mode         = mode;
    
    config.wavFolder    = [dataFolder,'wav\'];                          if ~exist(config.wavFolder); mkdir(config.wavFolder); end
    config.specFolder   = [dataFolder,'spectral\'];                     if ~exist(config.specFolder); mkdir(config.specFolder); end
    config.featFolder   = [dataFolder,'features_', architecture, '\'];  if ~exist(config.featFolder); mkdir(config.featFolder); end
    config.netFolder    = [dataFolder,'network_', architecture, '\'];   if ~exist(config.netFolder)
                                                                            mkdir([config.netFolder,'model\']);
                                                                            mkdir([config.netFolder,'predictions\']);
                                                                            mkdir([config.netFolder,'reconstruted\']);
                                                                        end
    
    allPcmFiles = dir([dataFolder, '*.pcm']);                           if isempty(allPcmFiles) || contains([allPcmFiles.name], 'error'); config.generatePCM = 0; end
    allWavFiles = dir([config.wavFolder, '*.wav']);                     if ~isempty(allWavFiles); config.generateWAV = 0; end
    allMatFiles = dir([config.specFolder, '*.mat']);                    if ~isempty(allMatFiles); config.generateFeat = 0; end
    if contains(config.mode,'train')
        config.statsFolder  = [dataFolder,'stats_', architecture, '\']; if ~exist(config.statsFolder) 
                                                                            mkdir(config.statsFolder);
                                                                        end
    else
        config.statsFolder  = uigetdir(pwd, 'Choose stats folder');
        config.modelFolder  = uigetdir(pwd, 'Choose model folder');
    end
    
    config.melFeatures              = 64;
    config.Fs                       = 16e3;
    config.fftWinSize_sec           = 32e-3;
    config.lmsDelay_samples         = 8e-3*config.Fs; 
    config.smoothingFactor          = 0.7; 
    config.architecture             = architecture;
    config.netDelay_frames          = 3;
    if strcmp(config.architecture, 'unet')
        config.netDelay_frames      = 30;
        config.fftWinSize_sec       = 20e-3;
    end
    config.nfft_samples             = config.Fs*config.fftWinSize_sec;

    save('config.mat', 'config');