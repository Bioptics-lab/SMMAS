function [neuronCenteri, coriAll, depthNeuron, rawtrace, pixels, centersAllview] = ...
    estimate_depth_other(stat, orderNeuron, coefXz, coefYz, viewi, views, view, move, factor, x0, y0, depthMin, depthMax, depthStep)
%ESTIMATE_DEPTH_OTHER Multi-view correlation depth search. view is 1-based.
if nargin < 9, factor = 2.5; end
if nargin < 10, x0 = 60; end
if nargin < 11, y0 = 60; end
if nargin < 12, depthMin = -40; end
if nargin < 13, depthMax = 40; end
if nargin < 14, depthStep = 2; end
view0 = view;
[h, w, nT, nViews] = size(views);
order = orderNeuron(:)';
nNeu = numel(order);
info = 1:4;
info1 = info(info ~= view0);
rawtrace = zeros(nNeu, nT);
neuronCenteri = zeros(nNeu, 3);
zGrid = depthMin:depthStep:depthMax;
nDepth = numel(zGrid);
coriAll = zeros(nDepth, nNeu);
pixels = struct('neuron_pixels_delta1', cell(nNeu, 1), 'neuron_pixels_delta2', []);
centersAllview = zeros(2, nNeu, 4);

for num = 1:nNeu
    st = stat(order(num));
    xpix = double(st.xpix(:)) - move(view0, 2);
    ypix = double(st.ypix(:)) - move(view0, 1);
    xpix = min(max(xpix, 0), w - 1);
    ypix = min(max(ypix, 0), h - 1);
    xi = min(max(round(xpix) + 1, 1), w); % 0-based suite2p -> 1-based MATLAB
    yi = min(max(round(ypix) + 1, 1), h);
    if ~isempty(xi)
        ind = sub2ind([h, w], yi, xi);
        plane = reshape(viewi, h * w, nT);
        rawtrace(num, :) = sum(plane(ind, :), 1);
        tmax = find(rawtrace(num, :) == max(rawtrace(num, :)), 1);
        wts = plane(ind, tmax);
        wsum = sum(wts);
        if wsum <= 0
            neuronCenteri(num, 1) = mean(xpix);
            neuronCenteri(num, 2) = mean(ypix);
        else
            neuronCenteri(num, 1) = sum(xpix .* wts) / wsum;
            neuronCenteri(num, 2) = sum(ypix .* wts) / wsum;
        end
    end
    d1 = xpix - neuronCenteri(num, 1);
    d2 = ypix - neuronCenteri(num, 2);
    pixels(num).neuron_pixels_delta1 = d1;
    pixels(num).neuron_pixels_delta2 = d2;
    coriZ = zeros(nDepth, 1);
    for iz = 1:nDepth
        nz = zGrid(iz);
        [dx, dy] = parallax_dx_dy(nz, view0 - 1, coefXz, coefYz, factor, x0, y0);
        refer = [neuronCenteri(num, 1) - dx, neuronCenteri(num, 2) - dy];
        coriThis = 0;
        for vo = info1
            [dx2, dy2] = parallax_dx_dy(nz, vo - 1, coefXz, coefYz, factor, x0, y0);
            guess = refer + [dx2, dy2];
            px = min(max(d1 + guess(1), 0), w - 1);
            py = min(max(d2 + guess(2), 0), h - 1);
            xj = min(max(round(px) + 1, 1), w);
            yj = min(max(round(py) + 1, 1), h);
            tr = mean_trace_vol(views, yj, xj, vo, nT);
            r = corrcoef_scalar(rawtrace(num, :), tr);
            if isfinite(r)
                coriThis = coriThis + r;
            end
        end
        coriZ(iz) = coriThis;
    end
    coriAll(:, num) = coriZ;
    finite = isfinite(coriZ);
    if ~any(finite)
        depthNeuronVal = 0;
    else
        peak = coriZ;
        peak(~finite) = -inf;
        hits = find(peak == max(peak));
        depthNeuronVal = mean(zGrid(hits));
    end
    [dx, dy] = parallax_dx_dy(depthNeuronVal, view0 - 1, coefXz, coefYz, factor, x0, y0);
    neuronCenteri(num, 1) = neuronCenteri(num, 1) - dx;
    neuronCenteri(num, 2) = neuronCenteri(num, 2) - dy;
    neuronCenteri(num, 3) = depthNeuronVal;
    centersAllview(1, num, view0) = neuronCenteri(num, 1);
    centersAllview(2, num, view0) = neuronCenteri(num, 2);
    for vn = 1:numel(info1)
        vo = info1(vn);
        [dx2, dy2] = parallax_dx_dy(depthNeuronVal, vo - 1, coefXz, coefYz, factor, x0, y0);
        centersAllview(1, num, vo) = neuronCenteri(num, 1) + dx2;
        centersAllview(2, num, vo) = neuronCenteri(num, 2) + dy2;
    end
end
depthNeuron = neuronCenteri(:, 3);
end

function tr = mean_trace_vol(views, yi, xi, v, nT)
if isempty(yi)
    tr = zeros(1, nT);
    return
end
ind = sub2ind([size(views, 1), size(views, 2)], yi, xi);
plane = reshape(views(:, :, :, v), [], nT);
tr = mean(plane(ind, :), 1);
end
