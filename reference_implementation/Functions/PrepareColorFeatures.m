function [colorFeatures, indNan] = PrepareColorFeatures(depth, images, refPos)

global param;

images = CropImg(images, param.depthBorder);

warpedImages = WarpAllImages(images, depth, refPos);

indNan = isnan(warpedImages);
warpedImages(indNan) = 0;


if (param.useGPU)
    warpedImages = gpuArray(warpedImages);
end

[h, w, ~] = size(depth);

refPos = reshape(refPos, 2, 1, 1, []);
colorFeatures = cat(3, depth, warpedImages, repmat(refPos(1, :, :, :)-1.5, h, w, 1, 1), repmat(refPos(2, :, :, :)-1.5, h, w, 1, 1));