function [melSTFT]  = applyMelScale(melFeatures, fullSTFT)
    
    if melFeatures ~= 64
        errordlg('Supports only mel64');
        melSTFT = [];
        return;
    end
    load('utils\Findex_64.mat');
    melSTFT = zeros(melFeatures,size(fullSTFT,2));
    for frame = 1:size(fullSTFT,2)
        for iter = 1:melFeatures
            melSTFT(iter,frame) = norm(fullSTFT(find(Findex == iter),frame));
        end
    end