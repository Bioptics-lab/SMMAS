function [fitLine, coefXz, coefYz, coefXy, x0, y0] = psf_fit(psfParts, zUm)
%PSF_FIT Independent x(z) and y(z) centroid trajectories. psfParts is Y,X,Z,view.
[nY, nX, nZ, nViews] = size(psfParts);
if nargin < 2 || isempty(zUm)
    zUm = (0:nZ-1) - (nZ - 1) / 2;
end
zUm = zUm(:)';
x0 = (nX + 1) / 2;
y0 = (nY + 1) / 2;
fitLine = zeros(2, nZ, nViews);
for v = 1:nViews
    for iz = 1:nZ
        image = psfParts(:, :, iz, v);
        [x, y] = largest_blob_centroid(image);
        fitLine(1, iz, v) = x;
        fitLine(2, iz, v) = y;
    end
end
coefXz = zeros(2, nViews);
coefYz = zeros(2, nViews);
coefXy = zeros(2, nViews);
for v = 1:nViews
    px = polyfit(zUm, fitLine(1, :, v), 1);
    py = polyfit(zUm, fitLine(2, :, v), 1);
    pxy = polyfit(fitLine(1, :, v), fitLine(2, :, v), 1);
    coefXz(:, v) = px(:);
    coefYz(:, v) = py(:);
    coefXy(:, v) = pxy(:);
end
end

function [x, y] = largest_blob_centroid(image)
mask = image > 0;
if ~any(mask(:))
    x = size(image, 2) / 2 + 0.5;
    y = size(image, 1) / 2 + 0.5;
    return
end
lab = bwlabel(mask, 8);
counts = accumarray(lab(:) + 1, 1);
counts(1) = 0;
[~, ind] = max(counts);
id = ind - 1;
[ys, xs] = find(lab == id);
x = mean(xs);
y = mean(ys);
end
