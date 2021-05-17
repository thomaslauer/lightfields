function error = TestDuringTraining(depthNet, colorNet)

global param;

sceneNames = param.testNames;
fid = fopen([param.trainNet, '/error.txt'], 'at');

numScenes = length(sceneNames);
error = 0;

for k = 1 : numScenes
    
    %%% read input data
    [images, depthFeatures, reference, refPos] = ReadTrainingData(sceneNames{k}, false);
    
    %%% evaluate the network and accumulate error
    [finalImg, ~, ~] = EvaluateSystem(depthNet, colorNet, images, refPos, true, depthFeatures, reference, true);
    
    finalImg = CropImg(finalImg, 10);
    reference = CropImg(reference, 10);
    
    curError = ComputePSNR(finalImg, reference);    
    error = error + curError / numScenes;
end

fprintf(fid, '%f\n', error);
fclose(fid);

error = gather(error);
