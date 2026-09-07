function nmask = generate_mask(neuronDictionary, coefXz, coefYz, xpsf, fov, factor, x0, y0)
%GENERATE_MASK Multi-view reconstruction mask from merged dictionary.
if nargin < 5, fov = 512; end
if nargin < 6, factor = 2.5; end
if nargin < 7, x0 = 60; end
if nargin < 8, y0 = 60; end
energyLen = 80;
xpsf = double(xpsf(:));
zi = linspace(1, numel(xpsf), energyLen);
xpsfi = interp1(xpsf, zi);
psfenergy1 = 1 ./ max(xpsfi(:), realmin('double'));
psfenergy1 = psfenergy1 / max(psfenergy1);
nmask = zeros(fov, fov, 4);
Info = 1:4;
for e = 1:numel(neuronDictionary)
    entry = neuronDictionary(e);
    cav = double(entry.center_allview);
    if ndims(cav) == 3
        cav = squeeze(cav);
    end
    center = double(entry.center);
    d1 = double(entry.pixels1(:));
    d2 = double(entry.pixels2(:));
    nz = center(3);
    nz1 = round((nz + 40) / 2) + 1;
    nz1 = min(max(nz1, 1), energyLen);
    w = psfenergy1(nz1);
    for vn = 1:4
        v0 = Info(vn) - 1;
        [dx, dy] = parallax_dx_dy(nz, v0, coefXz, coefYz, factor, x0, y0);
        cx = cav(1, vn) + dx;
        cy = cav(2, vn) + dy;
        px = min(max(round(d1 + cx), 1), fov);
        py = min(max(round(d2 + cy), 1), fov);
        for ii = 1:numel(px)
            nmask(py(ii), px(ii), vn) = w;
        end
    end
end
end
