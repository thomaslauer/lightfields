function TrainSystem(depthNet, colorNet)

global param;

testError = GetTestError(param.trainNet);
count = 0;
it = param.startIter + 1;

while (1)
    
    if (mod(it, param.printInfoIter) == 0)
        fprintf(repmat('\b', [1, count]));
        count = fprintf('Performing iteration %d', it);
    end
    
    %% main optimization
    [images, depthFeat, reference, refPos] = ReadTrainingData(param.trainingNames{1}, [], it);
    [~, depthRes, colorRes] = EvaluateSystem(depthNet, colorNet, images, refPos, true, depthFeat, reference);
    
    depthNet = UpdateNet(depthNet, depthRes, it);
    colorNet = UpdateNet(colorNet, colorRes, it);
    
    
    if (mod(it, param.testNetIter) == 0)
        %% save network
        [~, curNetName, ~] = GetFolderContent(param.trainNet, '.mat');
        fileName = sprintf('/Net-%06d.mat', it);
        save([param.trainNet, fileName], 'depthNet', 'colorNet');
        
        if (~isempty(curNetName))
            curNetName = curNetName{1};
            delete(curNetName);
        end
    
    
        %% perform validation
        countTest = fprintf('\nStarting the validation process\n');
        
        curError = TestDuringTraining(depthNet, colorNet);
        testError = [testError; curError];
        plot(1:length(testError), testError);
        title(sprintf('Current PSNR: %f', curError));
        drawnow;
        
        fprintf(repmat('\b', [1, countTest]));
    end
    
    it = it + 1;
end
