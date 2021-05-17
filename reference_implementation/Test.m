clearvars; clearvars -global; clc; close all; warning off;


%% Initialization
addpath('Functions');
addpath(genpath('./Libraries'));

sceneFolder = './Scenes';
resultFolder = './Results';

vl_setupnn();
InitParam();

%%% load the pre-trained networks
[depthNet, colorNet] = LoadNetworks();

%% Generate novel views for each scene

[sceneNames, scenePaths, numScenes] = GetFolderContent(sceneFolder);

for ns = 1 : numScenes
    fprintf('**********************************\n');
    fprintf('Working on the "%s" dataset\n', sceneNames{ns}(1:end-4));
    
    resultPath = [resultFolder, '/', sceneNames{ns}(1:end-4)];
    MakeDir([resultPath, '/Images']);
    
    fprintf('Loading input light field ...');
    [curFullLF, curInputLF] = ReadIllumImages(scenePaths{ns});
    fprintf(repmat('\b', 1, 3));
    fprintf('Done\n');
    fprintf('**********************************\n');
    
    fprintf('\nSynthesizing novel views\n');
    fprintf('--------------------------\n');
    SynthesizeNovelViews(depthNet, colorNet, curInputLF, curFullLF, resultPath);
    fprintf('\n\n\n');
end
warning on;