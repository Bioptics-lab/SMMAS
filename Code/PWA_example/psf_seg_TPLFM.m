function [psf_1]=psf_seg_TPLFM(ch1,center,zrange)

psf_1=zeros(241,241,size(zrange,2));

for ii=zrange
    planei=double(ch1(:,:,ii));
    planei=imresize(planei,[256 256]);
    stdd=std2(planei);
    planei= planei-mean(planei(:));
    planei(planei<3*stdd)=0;
    planei=planei(center(1)-120:center(1)+120,center(2)-120:center(2)+120);
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
