function [varargout] = convertTimeToSTFT(timeSignal, nfft, savingPath, name)

%% Doc:
% This function converts .wav files to STFT representation.
% INPUTS:
% timeSignal - time signal to be converted
% nfft, savingPath - taken from configuration, number of fft bins and path
% to save STFT
% name - extension used to identify the data (e.g., 'error','target', etc).

    signalSTFT = stft(timeSignal, nfft);
    ampSTFT = abs(signalSTFT);
    phaseSTFT = angle(signalSTFT);
    save([savingPath,'\amp_', num2str(name), '.mat'], 'ampSTFT','-v7.3');
    save([savingPath,'\phase_', num2str(name), '.mat'], 'phaseSTFT','-v7.3');
    
    varargout{1} = ampSTFT;
    varargout{2} = phaseSTFT;