function warpedImages = WarpAllImages(images, depth, refPos)

global inputView;

[h, w, c, numImages] = size(images);
numInputViews = length(inputView.Y);
images = gather(images); depth = gather(depth); refPos = gather(refPos);

warpedImages = zeros(h, w, c, numImages, 'single');

for i = 1 : numInputViews
    deltaY = inputView.Y(i) - refPos(1, :);
    deltaX = inputView.X(i) - refPos(2, :);
    warpedImages(:, :, (i-1)*3+1 : i*3, :) = WarpImages(depth, images(:, :, (i-1)*3+1:i*3, :), deltaY, deltaX);
end