function dzdx = PrepareColorFeaturesGrad(depth, images, refPos, curFeatures, indNan, dzdx)

delta = 0.01;

depthP = depth + delta;
[featuresP, indNanP] = PrepareColorFeatures(depthP, images, refPos);

grad = (featuresP - curFeatures) ./ delta .* dzdx;
tmp = grad(:, :, 2:end-2, :);
tmp(indNan | indNanP) = 0;
grad(:, :, 2:end-2, :) = tmp;

dzdx = sum(grad, 3);

