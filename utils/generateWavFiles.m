function [micSignal] = generateWavFiles(dataFolder, wavFolder, Fs)
    
%% Doc:
% This function converts .pcm from dataFolder to .wav files and saves them in wavFolder. Fs
% specifies the sample frequency of the data acquired.

    precision = 'int16';

    fid = fopen([dataFolder,'\error.pcm']); 
    if fid~=-1
    	AEC_NO_Post_process = int16(fread(fid, Inf, precision)); 
        AEC_NO_Post_process = double(AEC_NO_Post_process)/(2^15);
        audiowrite([wavFolder,'\error.wav'], AEC_NO_Post_process, Fs);
        fclose(fid);
        clear AEC_NO_Post_process;
    end

    fid = fopen([dataFolder,'\adapt.pcm']); 
    if fid~=-1
    	Adaptive_filter_out = int16(fread(fid, Inf, precision)); 
        Adaptive_filter_out = double(Adaptive_filter_out)/(2^15);
        audiowrite([wavFolder,'\adapt.wav'], Adaptive_filter_out, Fs);
        fclose(fid);
        clear Adaptive_filter_out;
    end
    
    fid = fopen([dataFolder,'\ref.pcm']); 
    if fid~=-1
    	farEndSignalOrig = int16(fread(fid, Inf, precision)); 
        farEndSignalOrig = double(farEndSignalOrig)/(2^15);
        audiowrite([wavFolder,'\ref.wav'], farEndSignalOrig, Fs);
        fclose(fid);
        clear farEndSignalOrig;
    end
    
    fid = fopen([dataFolder,'\mic.pcm']); 
    if fid~=-1
    	micSignal = int16(fread(fid, Inf, precision)); 
        micSignal = double(micSignal)/(2^15);
        audiowrite([wavFolder,'\mic.wav'], micSignal, Fs);
        fclose(fid);
        clear micSignal;
    end
    
    fid = fopen([dataFolder,'\target.pcm']); 
    if fid~=-1
    	target_signal = int16(fread(fid, Inf, precision)); 
        target_signal = double(target_signal)/(2^15);
        audiowrite([wavFolder,'\target.wav'], target_signal, Fs);
        fclose(fid);
        clear target_signal;
    end
