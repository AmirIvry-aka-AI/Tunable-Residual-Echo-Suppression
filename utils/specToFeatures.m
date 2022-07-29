function [] = specToFeatures(ampError, ampAdapt, config, varargin)
    
%% Doc:
% This function converts STFT representations to respective features.
% INPUTS:
% ampError - STFT amplitude of error channel
% ampAdapt - STFT amplitude of adaptive filter output channel
% config - system configuration
% varargin - (ampTarget), STFT amplitude of target channel, given training.

    statsFolder     = config.statsFolder;
    mode            = config.mode;
    architecture    = config.architecture;
    nfft            = config.nfft_samples;
    
    if strcmp(architecture, 'fcnn')

        melFeatures    = config.melFeatures;

        % STFT2MEL
        ampErrorMel    = applyMelScale(melFeatures, ampError);
        ampAdaptMel    = applyMelScale(melFeatures, ampAdapt);

        inputConcMel   = [ampErrorMel; ampAdaptMel];
        inputContMel   = [inputConcMel(:,3:end); inputConcMel(:,2:end-1); inputConcMel(:,1:end-2)];

        % Norm
        if strcmp(mode, 'train')
            stats.minInput  = min(inputContMel')';
            stats.maxInput  = max(inputContMel')';
        else
            stats           = structToData([statsFolder,'\stats.mat']); 
        end

        % Norm and prepare
        inputPrep           = applyScaling(inputContMel, stats.minInput, stats.maxInput, 'norm');

        % Handle target
        if ~isempty(varargin)
            ampTarget       = varargin{1};
            ampTargetMel    = applyMelScale(melFeatures, ampTarget);
            ampTargetMel    = ampTargetMel(:, 3:end);
            if strcmp(mode, 'train')
                stats.minTarget = min(ampTargetMel')';
                stats.maxTarget = max(ampTargetMel')';
                save([statsFolder,'stats.mat'],'stats','-v7.3');
            end
            targetPrep       = applyScaling(ampTargetMel, stats.minTarget, stats.maxTarget, 'norm');
        end        
    else

        inputConc   = [ampError; ampAdapt];

        % Norm
        if strcmp(mode, 'train')
            stats.minInput  = min(inputConc')';
            stats.maxInput  = max(inputConc')';
        else
            stats           = structToData([statsFolder,'\stats.mat']); 
        end

        % Norm and prepare
        inputNorm           = applyScaling(inputConc, stats.minInput, stats.maxInput, 'norm');
        inputPrep           = zeros(2, floor(nfft/2)+1, size(inputNorm,2));
        inputPrep(1,:,:)    = inputNorm(1:floor(nfft/2)+1, :);
        inputPrep(2,:,:)    = inputNorm(floor(nfft/2)+2:end, :);
        
        % Handle target
        if ~isempty(varargin)
            ampTarget       = varargin{1};
            if strcmp(mode, 'train')
                stats.minTarget = min(ampTarget')';
                stats.maxTarget = max(ampTarget')';
                save([statsFolder,'stats.mat'],'stats','-v7.3');
            end
            targetNorm          = applyScaling(ampTarget, stats.minTarget, stats.maxTarget, 'norm');
            targetPrep          = zeros(1, floor(nfft/2)+1, size(targetNorm,2));
            targetPrep(1,:,:)   = targetNorm;
        end 
        
    end
    
    if ~isempty(varargin)
        saveFeaturesList(inputPrep, config, targetPrep);
    else
        saveFeaturesList(inputPrep, config);
    end  