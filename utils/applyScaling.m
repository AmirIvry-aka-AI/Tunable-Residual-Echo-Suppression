function [scaledData] = applyScaling(unscaledData, statsMin, statsMax, mode)

dimUnscaledData = size(unscaledData, 2);
scale           = (repmat(statsMax,1,dimUnscaledData)-repmat(statsMin,1,dimUnscaledData));
bias            = repmat(statsMin,1,dimUnscaledData);

if strcmp(mode, 'norm')
    scaledData  = (unscaledData - bias)./scale;
else
    scaledData  = (unscaledData.*scale) + bias;
end