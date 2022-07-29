function [signal] = pcm2wav(signalPath, Fs)

    precision = 'int16';
    fid = fopen([signalPath, '.pcm']); 
    if fid~=-1
        signal = int16(fread(fid, Inf, precision)); 
        signal = double(signal)/(2^15);
        audiowrite([signalPath, '.wav'], signal, Fs);
        fclose(fid);
    end