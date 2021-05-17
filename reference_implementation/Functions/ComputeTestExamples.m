function [inImgs, inFeat, ref, refPos] = ComputeTestExamples(curFullLF, curInputLF)

global inputView;


[height, width, ~, ~, ~] = size(curInputLF);

%%%%%%%%%%%%% preparing input images %%%%%%%%%%%%%%%
inImgs = reshape(curInputLF, height, width, []);


%%%%%%%%%%%%% selecting random references %%%%%%%%%%
curRefInd.Y = 5; curRefInd.X = 5;
curRefPos.Y = GetImgPos(curRefInd.Y); curRefPos.X = GetImgPos(curRefInd.X);
    
fprintf('Working on reference (5, 5): ');

%%%%%%%%%%%%%%%%%%%%% preparing reference %%%%%%%%%%%%%%%%%%%%%%%%%%%
ref = curFullLF(:, :, :, curRefInd.Y, curRefInd.X);


%%%%%%%%%%%%%%%%%%%%% preparing features %%%%%%%%%%%%%%%%%%%%%%%%%
deltaViewY = inputView.Y - curRefPos.Y; 
deltaViewX = inputView.X - curRefPos.X;

inFeat = PrepareDepthFeatures(curInputLF, deltaViewY, deltaViewX);


%%%%%%%%%%%%%%%%%%%%%% preparing ref positions %%%%%%%%%%%%%%%%%%%
refPos = [curRefPos.Y; curRefPos.X];

fprintf(repmat('\b', 1, 5));
fprintf('Done\n');



