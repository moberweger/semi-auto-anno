% function to demonstrate how to use SIFT flow to register images across scenes
% For simplicity we use two satellite images from the Mars. These two images have different
% local appearances even though they were taken at the same location. Our job is to find
% the dense correspondence from one image to the other.

% Step 1. Load and downsample the images

load(sprintf('input_%d.mat', procid))

im1=im2double(in1);
im2=im2double(in2);

% Step 2. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

Sift1=dense_sift(im1,patchsize,gridspacing);
Sift2=dense_sift(im2,patchsize,gridspacing);

% Step 3. SIFT flow matching

% prepare the parameters
SIFTflowpara.alpha=2;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=3;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

tic;[vx,vy,energylist]=SIFTflowc2f(Sift1,Sift2,SIFTflowpara);toc
clear flow;
flow=zeros(size(im1,1), size(im1,2), 2);
flow(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,1)=vx;
flow(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,2)=vy;

% Step 4.  Visualize the matching results

Im2=im2(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
warpI2=ones(size(im2));
warpI2(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:)=warpImage(Im2,vx,vy);

save(sprintf('output_%d.mat', procid), 'flow', 'warpI2')
