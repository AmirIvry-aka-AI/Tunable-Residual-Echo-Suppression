function [fullPath] = wav2pcm(signalPath, signal)

    fid = fopen([signalPath, '.pcm'],'wb'); 
    fwrite(fid,signal*32767,'int16');
    fclose(fid);