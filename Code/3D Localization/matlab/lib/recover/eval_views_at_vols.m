function [rs, traces, rmax, score] = eval_views_at_vols(col, row, z, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp)
rs = nan(1, 4);
traces = zeros(4, vols.n_avg);
if isempty(seedFp)
    seedXs = []; seedYs = [];
else
    seedXs = seedFp.xs; seedYs = seedFp.ys;
end
for v0 = 0:3
    [cx, cy] = view_center_in_bin(col, row, z, p0, v0, moveI, coefXz, coefYz, factor, x0, y0);
    [xs, ys] = footprint_at_center(seedXs, seedYs, cx, cy, vols.ly, vols.lx);
    traces(v0 + 1, :) = vols_trace(vols, v0 + 1, xs, ys);
    r = corrcoef_scalar(traces(v0 + 1, :), trM2);
    if isfinite(r)
        rs(v0 + 1) = r;
    end
end
if any(isfinite(rs))
    rmax = max(rs(isfinite(rs)));
else
    rmax = -inf;
end
tmp = rs; tmp(~isfinite(tmp)) = 0;
score = sum(max(tmp, 0));
end
