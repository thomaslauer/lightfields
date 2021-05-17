function [depthNet, colorNet] = LoadNetworks(isTraining)

if(~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = false;
end

global param;

if (isTraining)
    netFolder = param.trainNet;
    [netName, ~, ~] = GetFolderContent(netFolder, '.mat');
    
    if (param.continue && ~isempty(netName))
        tokens = regexp(netName{1}, 'Net-([\d]+).mat', 'tokens');
        param.startIter = str2double(cell2mat(tokens{1}));
        load([netFolder, '/', netName{1}]);
    else
        param.continue = false;
        [depthNet, colorNet] = CreateNet();
    end
else
    netFolder = param.testNet;
    load([netFolder, '/Net']);
end


if (param.useGPU)
    depthNet = vl_simplenn_move(depthNet, 'gpu');
    depthNet = ConvertLayers(depthNet, 'gpu');
    colorNet = vl_simplenn_move(colorNet, 'gpu');
    colorNet = ConvertLayers(colorNet, 'gpu');
else
    depthNet = vl_simplenn_move(depthNet, 'cpu');
    depthNet = ConvertLayers(depthNet, 'cpu');
    colorNet = vl_simplenn_move(colorNet, 'cpu');
    colorNet = ConvertLayers(colorNet, 'cpu');
end