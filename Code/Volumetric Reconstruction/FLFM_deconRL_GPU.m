function [ObjRECON_t] = FLFM_deconRL_GPU(psf_path,data_path,savepath,Img_pixel,Psf_pixel,iter)
% Reconstruct a 3D volume from raw view images using
% Richardson-Lucy deconvolution on GPU. 
% SYSTEM REQUIREMENTS
% Memory: 8 GB RAM
% MATLAB: R2015a+

%% 100% GPU, only for 1X sampling

% Set workpath (do not cd: it breaks relative Data/ paths in the caller)
local_address=mfilename('fullpath');
[pathstr,namestr]=fileparts(local_address);
addpath(pathstr);

if ~exist(savepath, 'dir')
    mkdir(savepath);
end

% Load files for reconstruction

load([psf_path]); % load PSF: psf_lens with size (x, y, z, views)
load([data_path]); % load raw image: Data_raw with size (x, y, views, t)

stack_seg_wide = Data_raw(1:Img_pixel,1:Img_pixel,:,:); % only odd pixel sizes are supported
Info=[1,2,3,4]';  % views used for reconstruction; e.g. [1,2,4]' for views 1, 2, 4
timestacks=size(stack_seg_wide,4); % number of time points
image_seg=stack_seg_wide(:,:,:,1);
ObjRECON_t=zeros(size(image_seg,1),size(image_seg,2),size(psf_lens,3),timestacks,'single');

for timestack=1:timestacks
    
    image_seg=stack_seg_wide(:,:,:,timestack);
    
    % Select views and optionally upsample / downsample
    for ii=1:size(Info)
        Img(1:size(image_seg,1),1:size(image_seg,2),ii)=image_seg(:,:,Info(ii));
    end
    Img=single(Img);
    ximg=size(Img,1);
    yimg=size(Img,2);
    factor1=1;
    Img= imresize(Img,[factor1*ximg factor1*yimg],'cubic'); % change factor1 if upsampling
    
    % Reconstruction parameters
    ReconROIx=size(image_seg,1)/2-0.5;
    ReconROIy=size(image_seg,2)/2-0.5; % half-width of object
    FitROIx=size(image_seg,1)/2-0.5;
    FitROIy=size(image_seg,2)/2-0.5;
    % half-width of calibration area (>=ReconROI)
    ItN=iter; % number of iterations
    SNR=200; % estimated signal-to-noise ratio
    show_reconresult=1; % 1: show MIP of reconstruction every iteration; 0: off
    
    PSFROIx=size(psf_lens,1)/2-0.5;
    PSFROIy=size(psf_lens,2)/2-0.5;
    factor2=1;
    psf_lensn=zeros(factor2*Psf_pixel,factor2*Psf_pixel,size(psf_lens,3),size(Info,1),'single');
    for ii=1:size(Info)
        for nzz=1:size(psf_lens,3)
            psf_lensn(:,:,nzz,ii)=psf_lens(:,:,nzz,Info(ii));
        end
    end
    psf_lensn=gpuArray(psf_lensn);
    
    %%
    % Prepare for reconstruction
    Nz=size(psf_lensn,3);
    PSF_power=sum(psf_lensn(:));
    PSF_power_zn=sum(sum(sum(psf_lensn,1),2),4);
    PSF_power_zn=single(squeeze(repmat(PSF_power_zn,2*ReconROIx+1,2*ReconROIy+1,1,1)));
    PSF_power_zn=gather(PSF_power_zn);
    % Prepare for calibration
    x=[-FitROIx:FitROIx];
    y=[-FitROIy:FitROIy];
    [x y]=meshgrid(y,x);
    x=single(x);
    y=single(y);
    LensN=size(Info,1);
    
    for ii=1:LensN
        x_f_shift(:,:,ii)=x;
        y_f_shift(:,:,ii)=y;
        x_ff_shift(:,:,ii)=x;
        y_ff_shift(:,:,ii)=y;
    end
    
    %%
    
    ImgExp=gpuArray(single(Img));
    
    %%
    
    psf_lensn=single(psf_lensn);
    % Create variables
    ObjRecon=ones(2*ReconROIx+1,2*ReconROIy+1,Nz,'single','gpuArray'); % estimated object
    ImgEstROI=zeros(2*FitROIx+1,2*FitROIy+1,LensN,'single','gpuArray'); % estimated sub-image
    RatioROI=zeros(2*ReconROIx+1,2*ReconROIy+1,LensN,'single'); % sub-ratio
    RatioAvg=ObjRecon*0;
    ImgEst=Img*0;
    Ratio=ImgEst+1;
    RatioAvg=gather(RatioAvg);
    RatioROI_lens=zeros(2*ReconROIx+1,2*ReconROIy+1,LensN,'single');
    %%
    for iti=1:ItN
        tic;
        display(['iteration: ' num2str(iti)]);
        % Generate estimated sub-image
        psf_lensn=gpuArray(psf_lensn);
        ImgEstROI=gpuArray(ImgEstROI);
        ObjRecon=gpuArray(ObjRecon);
        
        for jj=1:LensN
            ImgEstROI(:,:,jj)=sum(real(ifft2(fft2(ifftshift(ifftshift(...
                padarray(psf_lensn(:,:,:,jj),[FitROIx-PSFROIx FitROIy-PSFROIy 0],0,'both')...
                ,1),2)).*fft2(padarray(ObjRecon,[FitROIx-ReconROIx FitROIy-ReconROIy],0,'both')))),3);
        end
        
        % Transform sub-image to aberrant sub-image and generate estimated image
        ImgEst=ImgEst*0;
        
        for ii=1:LensN
            % Interpolation removes aberration rather than applying a shift;
            % not needed for simulated data
            Img_lens=interp2(x,y,ImgEstROI(:,:,ii),x_ff_shift(:,:,ii),y_ff_shift(:,:,ii),'linear',0);
            
            Img_lens=gather(Img_lens);
            ImgEst(:,:,ii)=  ImgEst(:,:,ii)+Img_lens(FitROIx-ReconROIx+1:FitROIx+ReconROIx+1,FitROIy-ReconROIy+1:FitROIy+ReconROIy+1);
        end
        
        ImgEst=ImgEst/(PSF_power/Nz);
        Ratio=ImgExp./(ImgEst+mean(ImgEst(:))/SNR);
        Ratio_exp=Ratio;
        % Transform aberrant sub-ratio to calibrated sub-ratio
        RatioAvg=gpuArray(RatioAvg);
        RatioROI=gpuArray(RatioROI);
        RatioROI_lens=gpuArray(RatioROI_lens);
        
        for ii=1:LensN
            RatioROI_lens(:,:,ii)=interp2(x,y, Ratio_exp(:,:,ii)...
                ,x_f_shift(:,:,ii),y_f_shift(:,:,ii),'linear',0);
            
            RatioROI(:,:,ii)=RatioROI_lens(FitROIx-ReconROIx+1:FitROIx+ReconROIx+1,FitROIy-ReconROIy+1:FitROIy+ReconROIy+1,ii);
        end
        RatioAvg=RatioAvg*0;
        
        % Generate estimated object
        Ratio_exp=gather(Ratio_exp);
        RatioROI_lens=gather(RatioROI_lens);
        Ratio=gather(Ratio);
        ImgEst=gather(ImgEst);
        ImgEstROI=gather(ImgEstROI);
        ImgExp=gather(ImgExp);
        ObjRecon=gather(ObjRecon);
        
        for ii=1:LensN
            RatioAvg=RatioAvg+max(real(ifft2(fft2(repmat(RatioROI(:,:,ii),1,1,Nz))...
                .*conj(fft2(ifftshift(ifftshift(...
                padarray(psf_lensn(:,:,:,ii),[ReconROIx-PSFROIx ReconROIy-PSFROIy 0],0,'both')...
                ,1),2))))),0)./PSF_power_zn;
        end
        
        psf_power_zn=double(PSF_power_zn)/double(max(PSF_power_zn(:)));
        
        %%
        ObjRecon=ObjRecon.*RatioAvg;
        
        % Show MIP of reconstruction result
        if show_reconresult==1
            figure(2);
            subplot(1,3,1);
            imagesc(squeeze(max(ObjRecon(:,:,:),[],3)));
            title (['iteration ' num2str(iti) ' xy MIP']);
            xlabel ('x');
            ylabel ('y');
            axis equal;
            subplot(1,3,2);
            imagesc(squeeze(max(ObjRecon(:,:,:),[],2)));
            title(['iteration ' num2str(iti) ' yz MIP']);
            xlabel('z');
            ylabel('y');
            axis equal;
            subplot(1,3,3);
            imagesc(squeeze(max(ObjRecon(:,:,:),[],1)));
            title(['iteration ' num2str(iti) ' xz MIP']);
            xlabel('z');
            ylabel('x');
            axis equal;
            drawnow
        else
        end
        RatioROI=gather(RatioROI);
        RatioAvg=gather(RatioAvg);
        psf_lensn=gather(psf_lensn);
        toc;
    end
    ObjRECON_t(:,:,:,timestack)=gather(ObjRecon);
    
end
end
