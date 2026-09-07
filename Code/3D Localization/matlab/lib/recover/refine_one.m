function res = refine_one(m2, colSeed, rowSeed, zOld, tfm, vols, moveI, coefXz, coefYz, factor, x0, y0, cfg)
if nargin < 13, cfg = dataset_config(); end
trM2 = block_average_trace(m2.trace, vols.avg_bin, vols.n_avg);
seedFp = warped_m2_footprint(m2, tfm, vols.ly, vols.lx);
zSeed = zOld;
if ~isfinite(zSeed)
    zSeed = m2.z_um;
end
zLo = max(cfg.zMin, zSeed - cfg.zRefineHalf);
zHi = min(cfg.zMax, zSeed + cfg.zRefineHalf);
bestP0 = 0; bestScore = -inf;
for p0 = 0:3
    [~, ~, ~, score] = eval_views_at_vols(colSeed, rowSeed, zSeed, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
    if score > bestScore
        bestP0 = p0; bestScore = score;
    end
end
p0 = bestP0;
zGrid = zLo:cfg.zStep:zHi;
depthCurve = zeros(numel(zGrid), 2);
bestZAct = zSeed; bestSc = -inf;
for iz = 1:numel(zGrid)
    z = zGrid(iz);
    [~, ~, ~, score] = eval_views_at_vols(colSeed, rowSeed, z, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
    depthCurve(iz, :) = [z, score];
    if score > bestSc
        bestSc = score; bestZAct = z;
    end
end
bestCol = colSeed; bestRow = rowSeed; bestXy = bestSc;
for dcol = -6:2:6
    for drow = -6:2:6
        if dcol == 0 && drow == 0, continue; end
        [~, ~, ~, score] = eval_views_at_vols(colSeed + dcol, rowSeed + drow, bestZAct, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
        if score > bestXy
            bestXy = score; bestCol = colSeed + dcol; bestRow = rowSeed + drow;
        end
    end
end
[cx, cy, rsFp] = find_view_footprints(bestCol, bestRow, bestZAct, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp, cfg);
[zGeo, colGeo, rowGeo, rms] = fit_z_parallax(cx, cy, p0, moveI, coefXz, coefYz, factor, x0, y0, bestCol, bestRow, zLo, zHi, cfg);
if isfinite(zGeo) && isfinite(rms) && rms <= cfg.parallaxRmsMax
    bestZ = zGeo; bestCol = colGeo; bestRow = rowGeo; zSource = 'parallax';
else
    bestZ = bestZAct; zSource = 'activity';
end
[rs, tracesAvg, rmax, ~] = eval_views_at_vols(bestCol, bestRow, bestZ, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
nPass = sum(isfinite(rs) & rs >= cfg.refineRPass);
if isempty(seedFp)
    seedXs = []; seedYs = [];
else
    seedXs = seedFp.xs; seedYs = seedFp.ys;
end
[cxP, cyP] = view_center_in_bin(bestCol, bestRow, bestZ, p0, p0, moveI, coefXz, coefYz, factor, x0, y0);
[xsP, ysP] = footprint_at_center(seedXs, seedYs, cxP, cyP, vols.ly, vols.lx);
res.m2 = m2;
res.col = bestCol;
res.row = bestRow;
res.z2 = m2.z_um;
res.z_old = zOld;
res.best_z = bestZ;
res.z_activity = bestZAct;
res.z_parallax = zGeo;
res.parallax_rms_px = rms;
res.z_source = zSource;
res.best_r = rmax;
res.primary_v0 = p0;
res.rs = rs;
res.n_pass = nPass;
res.accepted = true;
res.traces = tracesAvg;
res.centers_x = cx;
res.centers_y = cy;
res.fp_rs = rsFp;
res.depth_curve = depthCurve;
res.fp_xs = xsP;
res.fp_ys = ysP;
res.fp_view = p0;
end

function [cxOut, cyOut, rs] = find_view_footprints(col0, row0, zHint, p0, vols, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp, cfg)
if isempty(seedFp)
    seedXs = []; seedYs = [];
else
    seedXs = seedFp.xs; seedYs = seedFp.ys;
end
cxOut = nan(1, 4); cyOut = nan(1, 4); rs = nan(1, 4);
xyLocal = cfg.xyLocal;
for v0 = 0:3
    [cx0, cy0] = view_center_in_bin(col0, row0, zHint, p0, v0, moveI, coefXz, coefYz, factor, x0, y0);
    bestR = -inf; bestCx = cx0; bestCy = cy0;
    for dcol = -xyLocal:2:xyLocal
        for drow = -xyLocal:2:xyLocal
            cx = cx0 + dcol; cy = cy0 + drow;
            [xs, ys] = footprint_at_center(seedXs, seedYs, cx, cy, vols.ly, vols.lx);
            r = corrcoef_scalar(vols_trace(vols, v0 + 1, xs, ys), trM2);
            if isfinite(r) && r > bestR
                bestR = r; bestCx = cx; bestCy = cy;
            end
        end
    end
    cx0 = bestCx; cy0 = bestCy;
    for dcol = -1:1
        for drow = -1:1
            if dcol == 0 && drow == 0, continue; end
            cx = cx0 + dcol; cy = cy0 + drow;
            [xs, ys] = footprint_at_center(seedXs, seedYs, cx, cy, vols.ly, vols.lx);
            r = corrcoef_scalar(vols_trace(vols, v0 + 1, xs, ys), trM2);
            if isfinite(r) && r > bestR
                bestR = r; bestCx = cx; bestCy = cy;
            end
        end
    end
    if bestR >= cfg.refineRPass
        cxOut(v0 + 1) = bestCx;
        cyOut(v0 + 1) = bestCy;
        rs(v0 + 1) = bestR;
    end
end
end

function [zBest, colBest, rowBest, rmsBest] = fit_z_parallax(cx, cy, p0, moveI, coefXz, coefYz, factor, x0, y0, col0, row0, zLo, zHi, cfg)
valid = isfinite(cx) & isfinite(cy);
if sum(valid) < 2
    zBest = nan; colBest = col0; rowBest = row0; rmsBest = nan;
    return
end
zBest = nan; colBest = col0; rowBest = row0; rmsBest = inf;
for z = zLo:cfg.zStep:zHi
    for dcol = [-2, 0, 2]
        for drow = [-2, 0, 2]
            col = col0 + dcol; row = row0 + drow;
            err2 = 0; n = 0;
            for v0 = 0:3
                if ~valid(v0 + 1), continue; end
                [px, py] = view_center_in_bin(col, row, z, p0, v0, moveI, coefXz, coefYz, factor, x0, y0);
                err2 = err2 + (px - cx(v0 + 1))^2 + (py - cy(v0 + 1))^2;
                n = n + 1;
            end
            rms = sqrt(err2 / max(n, 1));
            if rms < rmsBest
                zBest = z; colBest = col; rowBest = row; rmsBest = rms;
            end
        end
    end
end
end
