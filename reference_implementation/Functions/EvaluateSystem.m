function [finalImg, depthRes, colorRes] = EvaluateSystem(depthNet, colorNet, images, refPos, isTraining, depthFeatures, reference, isTestDuringTraining)

global param;
global inputView;

if (~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = false;
end

if (~exist('isTestDuringTraining', 'var') || isempty(isTestDuringTraining))
    isTestDuringTraining = false;
end

%% Estimating the depth (section 3.1)
if (~isTraining)
    fprintf('Estimating depth\n');
    fprintf('------------\n');

    fprintf('Extracting depth features ');
    dfTime = tic;
    deltaY = inputView.Y - refPos(1, :); 
    deltaX = inputView.X - refPos(2, :);
    depthFeatures = PrepareDepthFeatures(images, deltaY, deltaX);

    if (param.useGPU)
        depthFeatures = gpuArray(depthFeatures);
    end

    fprintf(repmat('\b', 1, 5));
    fprintf('Done in %f seconds\n', toc(dfTime));
end

if(~isTraining)
    fprintf('Evaluating depth network ...');
    dTime = tic;
end

depthRes = EvaluateNet(depthNet, depthFeatures, [], true);
depth = depthRes(end).x / (param.origAngRes-1);

if (~isTraining)
    fprintf(repmat('\b', 1, 3));
    fprintf('Done in %f seconds\n', toc(dTime));
end

%% Estimating the final color (section 3.2)
if(~isTraining)
    fprintf('Preparing color features ...');
    cfTime = tic;
    
    images = reshape(images, size(images, 1), size(images, 2), []);
end

[colorFeatures, indNan] = PrepareColorFeatures(depth, images, refPos);

if(~isTraining)
    fprintf(repmat('\b', 1, 3));
    fprintf('Done in %f seconds\n', toc(cfTime));
end

if(~isTraining)
    fprintf('Evaluating color network ...');
    cTime = tic;
end

colorRes = EvaluateNet(colorNet, colorFeatures, [], true);
finalImg = colorRes(end).x;

if(~isTraining)
    fprintf(repmat('\b', 1, 3));
    fprintf('Done in %f seconds\n', toc(cTime));
end



%% Backpropagation
if(isTraining && ~isTestDuringTraining)
    dzdx = vl_nnpdist(finalImg, reference, 2, 1, 'noRoot', true, 'aggregate', true) / numel(reference);
    
    colorRes(end).dzdx = dzdx;
    colorRes = EvaluateNet(colorNet, colorFeatures, colorRes, false);
    dzdx = colorRes(1).dzdx;
    
    dzdx = PrepareColorFeaturesGrad(depth, images, refPos, colorFeatures, indNan, dzdx);
    
    depthRes(end).dzdx = dzdx / (param.origAngRes-1);
    depthRes = EvaluateNet(depthNet, depthFeatures, depthRes, false);
    
end


