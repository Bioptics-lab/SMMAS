function [psf_1]=psf_seg_TPLFM(ch1,center,zrange)
% Segment a PSF stack around a given center and depth range.
% Example:
%   [psf_1n] = psf_seg_TPLFM(psf1, [164 145], 6:46);
% Plane 26 is treated as the center depth; [164 145] is the PSF center
% at that depth.

psf_1=zeros(141,141,size(zrange,2));

for ii=zrange
    
    planei=double(ch1(:,:,ii));
    
    stdd=std2(planei);
    
    planei= planei-mean( planei(:));
    
    planei(planei<3.5*stdd)=0;
    
    planei=planei(center(1)-70:center(1)+70,center(2)-70:center(2)+70);
    
    L=bwlabel(planei,4);

    stats=regionprops(L);

    Ar=cat(1,stats.Area);

    ind=find(Ar==max(Ar));

    planei(find(L~=ind))=0;
    
    psf_1(:,:,ii-min(zrange)+1)=planei;
    
    disp(['Processing frame:' num2str(ii)]);
    
end
