function output = WarpImages(disparity, input, delY, delX)

[h, w, numImages] = size(disparity);
[X, Y] = meshgrid(1:w, 1:h);
c = size(input, 3);

output = zeros(h, w, c, numImages, 'single');

for j = 1 : numImages
    for i = 1 : c
        
        curX = X + delX(j) * disparity(:, :, j);
        curY = Y + delY(j) * disparity(:, :, j);

        output(:, :, i, j) = interp2(X, Y, input(:, :, i, j), curX, curY, 'cubic', nan);
    end
end