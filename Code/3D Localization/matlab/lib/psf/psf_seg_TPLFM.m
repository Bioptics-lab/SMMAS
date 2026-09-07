function psf = psf_seg_TPLFM(ch1, center, zrange, cropHalfY, cropHalfX)
%PSF_SEG_TPLFM Crop, threshold, keep largest 4-connected blob per Z plane.
% center is 1-based (row, col); zrange is 1-based inclusive.
if nargin < 4, cropHalfY = 59; end
if nargin < 5, cropHalfX = 60; end
THRESH_STD = 3.5;
cy = round(center(1)) - 1;
cx = round(center(2)) - 1;
zIdx = zrange(:) - 1;
hy = cropHalfY;
hx = cropHalfX;
[nY, nX, ~] = size(ch1);
if cy - hy < 0 || cy + hy >= nY || cx - hx < 0 || cx + hx >= nX
    error('crop around center exceeds image');
end
nZ = numel(zIdx);
psf = zeros(2 * hy + 1, 2 * hx + 1, nZ);
for k = 1:nZ
    planei = double(ch1(:, :, zIdx(k) + 1));
    stdd = std(planei(:), 0);
    planei = planei - mean(planei(:));
    planei(planei < THRESH_STD * stdd) = 0;
    planei = planei(cy - hy + 1:cy + hy + 1, cx - hx + 1:cx + hx + 1);
    mask = planei > 0;
    if any(mask(:))
        lab = bwlabel(mask, 4);
        counts = accumarray(lab(:) + 1, 1);
        counts(1) = 0;
        [~, ind] = max(counts);
        planei(lab ~= (ind - 1)) = 0;
    end
    psf(:, :, k) = planei;
end
end
