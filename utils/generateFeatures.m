function [] = generateFeatures(config, totalSamples)

%% Doc:
% This function converts .wav files to STFT representation, and applies
% feature extraction tp it. It takes configurations from config and aligns
% the delay caused by the AEC system using totalSamples.

wavFolder           = config.wavFolder;
specFolder          = config.specFolder;
nfft_samples        = config.nfft_samples;
lmsDelay_samples    = config.lmsDelay_samples;

if exist([wavFolder,'\error.wav']) 
    error       = audioread([wavFolder,'\error.wav'],[lmsDelay_samples + 1,totalSamples]);
    ampError    = convertTimeToSTFT(error, nfft_samples, specFolder, 'error');
    clear error;
end
if exist([wavFolder,'\adapt.wav']) 
    adapt       = audioread([wavFolder,'\adapt.wav'],[lmsDelay_samples + 1,totalSamples]);
    ampAdapt    = convertTimeToSTFT(adapt, nfft_samples, specFolder, 'adapt');
    clear adapt;
end
if exist([wavFolder,'\ref.wav']) 
    ref         = audioread([wavFolder,'\ref.wav'],[1,totalSamples-lmsDelay_samples+1]);
    convertTimeToSTFT(ref, nfft_samples, specFolder, 'ref');
    clear ref;
end
if exist([wavFolder,'\mic.wav']) 
    mic         = audioread([wavFolder,'\mic.wav'],[1,totalSamples-lmsDelay_samples+1]);
    convertTimeToSTFT(mic, nfft_samples, specFolder, 'mic');
    clear mic;    
end
if exist([wavFolder,'\target.wav']) 
    target      = audioread([wavFolder,'\target.wav'],[1,totalSamples-lmsDelay_samples+1]);
    ampTarget   = convertTimeToSTFT(target, nfft_samples, specFolder, 'target');
    specToFeatures(ampError, ampAdapt, config, ampTarget);
else
    specToFeatures(ampError, ampAdapt, config);
end