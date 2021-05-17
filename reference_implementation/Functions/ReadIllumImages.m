function [fullLF, inputLF] = ReadIllumImages(scenePath)

numImgsX = 14;
numImgsY = 14;

%%% converting the extracted light field to a different format
inputImg = im2double(imread(scenePath));

h = size(inputImg, 1) / numImgsY;
w = size(inputImg, 2) / numImgsX;
fullLF = zeros(h, w, 3, numImgsY, numImgsX);
for ax = 1 : numImgsX
    for ay = 1 : numImgsY
        fullLF(:, :, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
    end
end

if (h == 375 && w == 540)
    fullLF = padarray(fullLF, [1, 1], 0, 'post');
end

if (h == 375 && w == 541)
    fullLF = padarray(fullLF, [1, 0], 0, 'post');
end

fullLF = fullLF(:, :, :, 4:11, 4:11); % we only take the 8 middle images
inputLF = fullLF(:, :, :, [1, 8], [1, 8]); % we use the four corner sub-aperture images as our input

