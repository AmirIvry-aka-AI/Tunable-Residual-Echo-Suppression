function [D] = structToData(pathtoStruct)

holder  = load(pathtoStruct); 
fNames  = fieldnames(holder); 
D       = getfield(holder, fNames{1});
