function PrepareTestData()

global param;

sceneFolder = param.testScenes;
outputFolder = param.testData;

[sceneNames, scenePaths, numScenes] = GetFolderContent(sceneFolder, '.png');

for ns = 1 : numScenes
    
    curOutputName = [outputFolder, '/', sceneNames{ns}(1:end-4)];
    
    fprintf('**********************************\n');
    fprintf('Working on the "%s" dataset (%d of %d)\n', sceneNames{ns}(1:end-4), ns, numScenes);
    
    fprintf('Loading input light field ...');
    [curFullLF, curInputLF] = ReadIllumImages(scenePaths{ns});
    fprintf(repmat('\b', 1, 3));
    fprintf('Done\n');
    fprintf('**********************************\n');    
    
    fprintf('\n\nPreparing test examples\n');
    fprintf('------------------------------\n');
    [pInImgs, pInFeat, pRef, refPos] = ComputeTestExamples(curFullLF, curInputLF);
    
    fprintf('\n\nWriting test examples\n\n');
    WriteTestExamples(pInImgs, pInFeat, pRef, refPos, curOutputName);
    
end