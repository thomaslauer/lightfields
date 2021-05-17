function InitParam()

global param;

%%% DO NOT CHANGE ANY PARAMETER, UNLESS YOU KNOW EXACTLY WHAT EACH PARAMETER DOES

param.depthResolution = 100; % number of depth levels (see Eq. 5)
param.numDepthFeatureChannels = param.depthResolution * 2;
param.deltaDisparity = 21; % the range of disparities (see Eq. 5)
param.origAngRes = 8; % original angular resolution
param.depthBorder = 6; % the number of pixels that are reduce due to the convolution process in the depth network. This border can be avoided if the networks are padded appropriately.
param.colorBorder = 6;  % same as above, for the color network.
param.testNet = 'TrainedNetworks';

%%% here, we set the desired novel views and indexes of the input views.
global novelView;
global inputView;

[novelView.X, novelView.Y] = meshgrid(1:1/7:2, 1:1/7:2);
[inputView.X, inputView.Y] = meshgrid(1:2, 1:2);
novelView.Y = novelView.Y'; novelView.X = novelView.X';
novelView.Y = novelView.Y(:); novelView.X = novelView.X(:);
inputView.Y = inputView.Y(:); inputView.X = inputView.X(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% If you have compiled MatConvNet with GPU and CuDNN supports, then leave
%%% these parameters as is. Otherwise change them appropriately.
param.useGPU = true; 
param.gpuMethod = 'Cudnn';%'NoCudnn';%


%%%%%%%%%%%%%%%%%% Training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.height = 376;
param.width = 541;

param.patchSize = 60;
param.stride = 16;
param.numRefs = 4; % number of reference images selected randomly on each light field
param.cropSizeTraining = 70; % we crop the boundaries to avoid artifacts in the training
param.batchSize = 10; % we found batch size of 10 is faster and gives the same results

param.cropHeight = param.height - 2 * param.cropSizeTraining;
param.cropWidth = param.width - 2 * param.cropSizeTraining;

param.trainingScenes = 'TrainingData/Training/';
param.trainingData = 'TrainingData/Training/';
[~, param.trainingNames, ~] = GetFolderContent(param.trainingData, '.h5');

param.testScenes = 'TrainingData/Test/';
param.testData = 'TrainingData/Test/';
[~, param.testNames, ~] = GetFolderContent(param.testData, '.h5');

param.trainNet = 'TrainingData';


param.continue = true;
param.startIter = 0;

param.testNetIter = 100;
param.printInfoIter = 5;


%%% ADAM parameters
param.alpha = 0.0001;
param.beta1 = 0.9;
param.beta2 = 0.999;
param.eps = 1e-8;