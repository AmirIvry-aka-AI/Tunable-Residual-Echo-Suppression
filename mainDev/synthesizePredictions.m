function [] = synthesizePredictions(config)

%% Doc:
% This function receives the prediction from the network and synthesizes
% it, then saves it to folders specified by config.

    architecture        = config.architecture;
    netDelay_frames     = config.netDelay_frames;
    predictionsFolder   = [config.netFolder,'predictions\'];
    reconFolder         = [config.netFolder,'reconstruted\'];
    featFolder          = config.featFolder;
    specFolder          = config.specFolder;
    statsFolder         = config.statsFolder;
    Fs                  = config.Fs;
    stats               = structToData([statsFolder, '\stats.mat']);
    minTarget           = stats.minTarget; 
    maxTarget           = stats.maxTarget;
    minInput            = stats.minInput; 
    maxInput            = stats.maxInput;
    
    startingPosition    = netDelay_frames;

    if strcmp(architecture, 'fcnn')
        
        Findex              = structToData('Findex_64.mat');
        melFeatures         = config.melFeatures;
        smoothingFactor     = config.smoothingFactor; 
        melPred             = double(readNPY([predictionsFolder,'prediction.npy']))'; 
        melPredDenorm       = applyScaling(melPred, minTarget, maxTarget, 'denorm'); 

        melInput            = structToData([featFolder,'input.mat']);
        melInputDenorm      = applyScaling(melInput(1:melFeatures, :), minInput(1:melFeatures), maxInput(1:melFeatures), 'denorm'); 

        gain                = melPredDenorm./melInputDenorm;
        gainExtend          = zeros(size(Findex,2),size(gain,2));
        for frame = 1:size(gain,2)
            for iter = 1:length(Findex)
                gainExtend(iter, frame) = gain(Findex(iter),frame);
            end
        end

        gainSmooth          = zeros(size(gainExtend));
        gainSmooth(:, 1)    = gainExtend(:,1);
        for iter = 2:size(gainSmooth, 2)
            gainSmooth(:, iter) = ...
                smoothingFactor*gainSmooth(:, iter-1) + (1-smoothingFactor)*gainExtend(:, iter);
        end
        gainExtend          = gainSmooth;

        stftErrorAmp        = structToData([specFolder,'amp_error.mat']);
        stftErrorPhase      = structToData([specFolder,'phase_error.mat']);
        stftPred            = stftErrorAmp(:, startingPosition:end).*gainExtend;
        timePred            = istft(stftPred.*exp(1i*stftErrorPhase(:, startingPosition:end)));

        audiowrite([reconFolder,'recon.wav'], timePred, Fs);
        
    else
        
        stftPredAmp         = double(readNPY([predictionsFolder,'prediction.npy']))'; 
        stftPredAmpDenorm   = applyScaling(stftPredAmp, minTarget, maxTarget, 'denorm'); 
        stftErrorPhase      = structToData([specFolder,'phase_error.mat']);
        timePred            = istft(stftPredAmpDenorm.*exp(1i*stftErrorPhase(:, startingPosition:end)));

        audiowrite([reconFolder,'recon.wav'], timePred, Fs);
        
    end