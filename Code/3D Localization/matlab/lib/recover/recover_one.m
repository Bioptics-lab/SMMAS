function res = recover_one(m2, tfm, mems, ly, lx, nf, moveI, coefXz, coefYz, factor, x0, y0, cfg)
if nargin < 13, cfg = dataset_config(); end
x2 = m2.x; y2 = m2.y; z2 = m2.z_um;
[col0, row0] = inverse_xy(x2, y2, tfm);
trFull = double(m2.trace(:))';
nT = min(nf, numel(trFull));
frameIdx = 1:cfg.tStrideRecover:nT;
trM2 = trFull(frameIdx);
seedFp = warped_m2_footprint(m2, tfm, ly, lx);
boxes = search_boxes(col0, row0, z2, moveI, coefXz, coefYz, factor, x0, y0, ly, lx, seedFp, cfg);
crops = make_crop_views(mems, boxes, frameIdx);

bestP0 = 0; bestScore = -inf;
for p0 = 0:3
    [~, ~, ~, score] = eval_views_at(col0, row0, z2, p0, crops, ly, lx, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
    if score > bestScore
        bestP0 = p0; bestScore = score;
    end
end
p0 = bestP0;
zGrid = (z2 - cfg.zSearchHalf):cfg.zStep:(z2 + cfg.zSearchHalf);
depthCurve = zeros(numel(zGrid), 2);
bestZ = z2; bestZs = -inf;
for iz = 1:numel(zGrid)
    z = zGrid(iz);
    [~, ~, ~, score] = eval_views_at(col0, row0, z, p0, crops, ly, lx, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
    depthCurve(iz, :) = [z, score];
    if score > bestZs
        bestZs = score; bestZ = z;
    end
end
bestCol = col0; bestRow = row0; bestXy = bestZs;
for dcol = -cfg.xyCoarseHalf:3:cfg.xyCoarseHalf
    for drow = -cfg.xyCoarseHalf:3:cfg.xyCoarseHalf
        if dcol == 0 && drow == 0, continue; end
        [~, ~, ~, score] = eval_views_at(col0 + dcol, row0 + drow, bestZ, p0, crops, ly, lx, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
        if score > bestXy
            bestXy = score; bestCol = col0 + dcol; bestRow = row0 + drow;
        end
    end
end
bc0 = bestCol; br0 = bestRow;
for dcol = -cfg.xyFineHalf:cfg.xyFineHalf
    for drow = -cfg.xyFineHalf:cfg.xyFineHalf
        if dcol == 0 && drow == 0, continue; end
        [~, ~, ~, score] = eval_views_at(bc0 + dcol, br0 + drow, bestZ, p0, crops, ly, lx, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
        if score > bestXy
            bestXy = score; bestCol = bc0 + dcol; bestRow = br0 + drow;
        end
    end
end
[rs, traces, rmax, ~] = eval_views_at(bestCol, bestRow, bestZ, p0, crops, ly, lx, trM2, moveI, coefXz, coefYz, factor, x0, y0, seedFp);
nPass = sum(isfinite(rs) & rs >= cfg.rPass);
accepted = isfinite(rmax) && rmax >= cfg.rBest && nPass >= cfg.nPass;
res.m2 = m2;
res.col = bestCol;
res.row = bestRow;
res.z2 = z2;
res.best_r = rmax;
res.best_z = bestZ;
res.primary_v0 = p0;
res.rs = rs;
res.n_pass = nPass;
res.accepted = accepted;
res.traces = traces;
res.frame_idx = frameIdx;
res.tr_m2_stride = trM2;
res.depth_curve = depthCurve;
res.used_warped_fp = ~isempty(seedFp);
end
