function [] = prepareDataForNet(config)

melFolder       = [config.featFolder,'mel\'];
errorFolder     = [config.featFolder,'error\'];
prepFolder      = [config.netFolder,'prepared\'];
if strcmp(config.dataToPrepare_sec, 'all')
    batchesUsed = length(dir([errorFolder,'*.mat']));
else
    batchesUsed = max(1, floor(config.dataToPrepare_sec/config.batchSplit_sec));
end
mode            = config.mode;

beg             = 1;
for batch       = 1:batchesUsed
    batchString = num2str(batch);
    melInput    = structToData([melFolder,'input_',batchString,'.mat']);
    fin         = size(melInput,2);
    
    input(:, beg:beg+fin-1)      = melInput;
    if strcmp(mode, 'train')
        melTarget                = structToData([melFolder,'target_',batchString,'.mat']);
        target(:, beg:beg+fin-1) = melTarget;
    end
    
    beg         = beg+fin;
end

save([prepFolder,'input.mat'], 'input', '-v7.3');
if strcmp(mode, 'train')
    save([prepFolder,'target.mat'], 'target', '-v7.3');
end