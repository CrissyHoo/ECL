function psnr=compute_psnr(img1,img2,scale)
imdff = double(img1) - double(img2);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));
psnr=20*log10(255/rmse);