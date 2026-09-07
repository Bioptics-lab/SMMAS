function [psf_1,coef_psf_xz,coef_psf_yz]=psf_seg_TPLFM_fit(ch1,center,zrange)

psf_1=zeros(141,143,size(zrange,2));

for ii=zrange
    planei=double(ch1(:,:,ii));
    planei=imresize(planei,[256 256]);
    stdd=std2(planei);
    planei= planei-mean(planei(:));
    planei(planei<3*stdd)=0;
    planei=planei(center(1)-70:center(1)+70,center(2)-71:center(2)+71);
    L=bwlabel(planei);
    stats=regionprops(L);
    Ar=cat(1,stats.Area);
    ind=find(Ar==max(Ar));
    planei(find(L~=ind))=0;
    psf_1(:,:,ii-min(zrange)+1)=planei;
    X=stats(ind).Centroid;
    fit_line(1:2,ii)=X(1,:);
end

Reconstruct_range = max(zrange)-min(zrange);
z=(-Reconstruct_range/2):1:(Reconstruct_range/2);

% coef_psf_xz=polyfit(fit_line(1,zrange), z, 1);
% coef_psf_yz=polyfit(fit_line(2,zrange), z, 1);
coef_psf_xz= - polyfit(z,fit_line(1,zrange),1);
coef_psf_yz= - polyfit(z,fit_line(2,zrange),1);