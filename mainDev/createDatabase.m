function [] = createDatabase(dataFolder, config)

%% Doc:
% This function creates all data structures in subfolders of dataFolder
% specified by config, including .pcm, .wav, STFT, and relevant features
% INPUTS:
% dataFolder - the folder in which the .pcm files are located in
% config - System configuration. A struct with relevant fields for system operation

    %% Run AEC AND SAVE TO PCM
    if config.generatePCM
        ecancel_demo_extend(dataFolder);
    end

    %% PCM TO WAV
    if config.generateWAV
        generateWavFiles(dataFolder, config.wavFolder, config.Fs);
    end

    %% WAV TO STFT AND TO FEATURES
    if config.generateFeat
        info = audioinfo([config.wavFolder,'mic.wav']);
        generateFeatures(config, info.TotalSamples);
    end