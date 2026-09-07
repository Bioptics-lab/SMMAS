function boxes = search_boxes(col0, row0, z2, moveI, coefXz, coefYz, factor, x0, y0, ly, lx, seedFp, cfg)
if nargin < 13, cfg = dataset_config(); end
if ~isempty(seedFp)
    fpHx = max(abs(double(seedFp.xs) - mean(double(seedFp.xs))));
    fpHy = max(abs(double(seedFp.ys) - mean(double(seedFp.ys))));
else
    fpHx = cfg.diskRadius;
    fpHy = cfg.diskRadius;
end
pad = max(fpHx, fpHy) + cfg.xyCoarseHalf + 3.0;
boxes = zeros(4, 4);
for v0 = 0:3
    cxMin = inf; cxMax = -inf; cyMin = inf; cyMax = -inf;
    for z = [z2 - cfg.zSearchHalf, z2 + cfg.zSearchHalf]
        for p0 = 0:3
            [cx, cy] = view_center_in_bin(col0, row0, z, p0, v0, moveI, coefXz, coefYz, factor, x0, y0);
            cxMin = min(cxMin, cx); cxMax = max(cxMax, cx);
            cyMin = min(cyMin, cy); cyMax = max(cyMax, cy);
        end
    end
    x0b = max(0, floor(cxMin - pad));
    x1b = min(lx, ceil(cxMax + pad + 1));
    y0b = max(0, floor(cyMin - pad));
    y1b = min(ly, ceil(cyMax + pad + 1));
    boxes(v0 + 1, :) = [y0b, y1b, x0b, x1b];
end
end
