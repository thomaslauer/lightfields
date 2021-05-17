function WriteError(estimated, reference, resultPath)

curPSNR = ComputePSNR(estimated, reference);
curSSIM = ssim(estimated, reference);

fid = fopen([resultPath, '/ObjectiveQuality.txt'], 'wt');
fprintf(fid, 'PSNR: %3.2f\n', curPSNR);
fprintf(fid, 'SSIM: %1.3f\n', curSSIM);
fclose(fid);