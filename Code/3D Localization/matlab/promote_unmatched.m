function promote_unmatched(cfg)
%PROMOTE_UNMATCHED Append unmatched recon ROIs into MATLAB working view suite2p copies.
% Does not rewrite original Views/.
if nargin < 1, cfg = dataset_config(); end
matchDir = resolve_match_dir(cfg);
matchCsv = fullfile(matchDir, 'matches.csv');
unmatchCsv = fullfile(matchDir, 'unmatched_m2.csv');
if ~exist(matchCsv, 'file')
    fprintf('no matches.csv under %s; skip promote\n', matchDir);
    return
end
if ~exist(unmatchCsv, 'file')
    fprintf('no unmatched_m2.csv; nothing to promote\n');
    return
end
if ~views_movies_available(cfg)
    fprintf('no TIFF/data.bin; skip promote (will not rewrite original Views/)\n');
    return
end

matches = load_matches(matchCsv);
[trusted, ~] = classify_matches(matches, cfg);
trustedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(trusted)
    trustedKeys(sprintf('%d_%d', trusted(i).m2_plane, trusted(i).m2_roi)) = true;
end
umRows = read_csv(unmatchCsv);
candidates = [];
for i = 1:numel(umRows)
    key = sprintf('%d_%d', umRows(i).m2_plane, umRows(i).m2_roi);
    if ~isKey(trustedKeys, key)
        if isempty(candidates)
            candidates = umRows(i);
        else
            candidates(end+1) = umRows(i); %#ok<AGROW>
        end
    end
end
fprintf('unmatched=%d  already trusted=%d  to try=%d\n', numel(umRows), trustedKeys.Count, numel(candidates));
if isempty(candidates)
    fprintf('no unmatched m2 left to promote\n');
    return
end

m2Pool = load_method2(cfg);
m2Map = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(m2Pool)
    m2Map(sprintf('%d_%d', m2Pool(i).plane_id, m2Pool(i).roi_idx)) = i;
end
tfm = load_xy_tfm(matchDir);
calib = load_psf_coef(cfg);
coefXz = calib.coef_psf_xz; coefYz = calib.coef_psf_yz;
factor = calib.factor; x0 = calib.x0; y0 = calib.y0;
moveI = round(compute_move(calib.centers_1based, cfg.moveDivisor));
[mems, ly, lx, nf] = open_view_memmaps(cfg);
for v = 1:4
    ensure_working_view(v, cfg);
end

mapPath = fullfile(cfg.outDir, 'views_suite2p_from_recon', 'appended_unmatched.json');
ensure_dir(fileparts(mapPath));
mp = struct('appended', {{}});
promotedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
nSkipR = 0;
umFields = fieldnames(umRows);

for k = 1:numel(candidates)
    key = sprintf('%d_%d', candidates(k).m2_plane, candidates(k).m2_roi);
    if ~isKey(m2Map, key)
        fprintf('  skip %s (not in pool)\n', key);
        continue
    end
    m2 = m2Pool(m2Map(key));
    fprintf('  [%d/%d] p%d/roi%d ...\n', k, numel(candidates), m2.plane_id, m2.roi_idx);
    seed = warped_m2_footprint(m2, tfm, ly, lx);
    if isempty(seed)
        fprintf('    no warped footprint\n');
        continue
    end
    [col0, row0] = inverse_xy(m2.x, m2.y, tfm);
    res = recover_one(m2, tfm, mems, ly, lx, nf, moveI, coefXz, coefYz, factor, x0, y0, cfg);
    zPlace = res.best_z;
    p0Place = res.primary_v0;
    if ~isfinite(res.best_r) || res.best_r < cfg.trustRMin
        nSkipR = nSkipR + 1;
        continue
    end
    for v0 = 0:3
        [cx, cy] = view_center_in_bin(col0, row0, zPlace, p0Place, v0, moveI, coefXz, coefYz, factor, x0, y0);
        [xs, ys] = footprint_at_center(seed.xs, seed.ys, cx, cy, ly, lx);
        append_working_roi(v0 + 1, xs, ys, mems{v0 + 1}, nf, cfg);
    end
    col = res.col; rowY = res.row; z = res.best_z;
    [x1m, y1m] = apply_xy(col, rowY, tfm);
    dXy = hypot(x1m - m2.x, y1m - m2.y);
    dZ = abs(z - m2.z_um);
    corr = res.best_r;
    trustedNow = isfinite(corr) && corr >= cfg.trustRMin && dXy <= cfg.trustDXyPx && dZ <= cfg.trustDZUm;
    fprintf('    z=%.1f xy=(%.1f, %.1f) corr=%.3f dxy=%.2f dz=%.1f -> %s\n', ...
        z, col, rowY, corr, dXy, dZ, ternary(trustedNow, 'TRUSTED', 'keep unmatched'));
    rec = struct('m2_plane', m2.plane_id, 'm2_roi', m2.roi_idx, 'promoted', trustedNow, ...
        'corr', corr, 'd_xy_px', dXy, 'd_z_um', dZ);
    mp.appended{end+1} = rec; %#ok<AGROW>
    if ~trustedNow
        continue
    end
    e = build_synthetic_entry(res, coefXz, coefYz, factor, x0, y0);
    e.source = 'matched_trusted';
    m1Idx = append_method1_files(e, col, rowY, z, corr, res.primary_v0 + 1, dXy, dZ, x1m, y1m, m2, matchCsv, cfg);
    rec.m1_idx = m1Idx;
    promotedKeys(key) = true;
    keepUm = true(numel(umRows), 1);
    for u = 1:numel(umRows)
        uk = sprintf('%d_%d', umRows(u).m2_plane, umRows(u).m2_roi);
        if isKey(promotedKeys, uk)
            keepUm(u) = false;
        end
    end
    umRows = umRows(keepUm);
    write_csv(unmatchCsv, umRows, umFields);
end
write_json(mapPath, mp);
fprintf('promoted %d  skipped_low_r %d  unmatched left %d\n', promotedKeys.Count, nSkipR, numel(umRows));
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

function dst = ensure_working_view(v, cfg)
dst = fullfile(cfg.outDir, 'views_suite2p_working', sprintf('view_%d', v), 'suite2p', 'plane0');
if exist(fullfile(dst, 'F.npy'), 'file')
    return
end
src = view_plane_dir(v, cfg);
ensure_dir(dst);
copyfile(fullfile(src, 'F.npy'), fullfile(dst, 'F.npy'));
copyfile(fullfile(src, 'Fneu.npy'), fullfile(dst, 'Fneu.npy'));
copyfile(fullfile(src, 'iscell.npy'), fullfile(dst, 'iscell.npy'));
if exist(fullfile(src, 's2p_matlab'), 'dir')
    copyfile(fullfile(src, 's2p_matlab'), fullfile(dst, 's2p_matlab'));
end
fprintf('copied view%d suite2p -> %s (original Views/ untouched)\n', v, dst);
end

function append_working_roi(v, xs, ys, mm, nf, cfg)
dst = fullfile(cfg.outDir, 'views_suite2p_working', sprintf('view_%d', v), 'suite2p', 'plane0');
F = double(readNPY(fullfile(dst, 'F.npy')));
Fneu = double(readNPY(fullfile(dst, 'Fneu.npy')));
iscell = double(readNPY(fullfile(dst, 'iscell.npy')));
frow = zeros(1, nf);
chunk = 80;
xs = double(xs(:)); ys = double(ys(:));
for s = 1:chunk:nf
    e = min(s + chunk - 1, nf);
    acc = zeros(1, e - s + 1);
    for t = s:e
        fr = read_bin_frame(mm, t);
        pix = 0; n = 0;
        for p = 1:numel(xs)
            yy = ys(p) + 1; xx = xs(p) + 1;
            if yy >= 1 && yy <= size(fr, 1) && xx >= 1 && xx <= size(fr, 2)
                pix = pix + fr(yy, xx);
                n = n + 1;
            end
        end
        if n > 0
            acc(t - s + 1) = pix / n;
        end
    end
    frow(s:e) = acc;
end
F = [F; frow];
zrow = zeros(1, size(F, 2));
Fneu = [Fneu; zrow];
if size(iscell, 2) >= 2
    iscell = [iscell; 1, 1];
else
    iscell = [iscell(:); 1];
end
writeNPY(single(F), fullfile(dst, 'F.npy'));
writeNPY(single(Fneu), fullfile(dst, 'Fneu.npy'));
writeNPY(iscell, fullfile(dst, 'iscell.npy'));
meta = fullfile(dst, 's2p_matlab');
if exist(meta, 'dir')
    xpix = readNPY(fullfile(meta, 'xpix.npy'));
    ypix = readNPY(fullfile(meta, 'ypix.npy'));
    lam = readNPY(fullfile(meta, 'lam.npy'));
    off = readNPY(fullfile(meta, 'pix_offset.npy'));
    cnt = readNPY(fullfile(meta, 'pix_count.npy'));
    med = readNPY(fullfile(meta, 'med.npy'));
    writeNPY([xpix(:); xs], fullfile(meta, 'xpix.npy'));
    writeNPY([ypix(:); ys], fullfile(meta, 'ypix.npy'));
    writeNPY([lam(:); ones(numel(xs), 1)], fullfile(meta, 'lam.npy'));
    writeNPY(int32([off(:); numel(xpix)]), fullfile(meta, 'pix_offset.npy'));
    writeNPY(int32([cnt(:); numel(xs)]), fullfile(meta, 'pix_count.npy'));
    writeNPY([med; mean(ys), mean(xs)], fullfile(meta, 'med.npy'));
    opsFile = fullfile(meta, 'ops.json');
    opsJ = jsondecode(fileread(opsFile));
    opsJ.nroi = opsJ.nroi + 1;
    write_json(opsFile, opsJ);
end
end

function m1Idx = append_method1_files(e, col, rowY, z, corr, pView, dXy, dZ, x1m, y1m, m2, matchCsv, cfg)
dictPath = fullfile(cfg.outDir, 'neuron_dictionary.mat');
d = load(dictPath);
merged = d.merged;
m1Idx = numel(merged); % 0-based after append
e.center = [col, rowY, z];
e.primary_view = pView;
e.peak_corr = corr;
e.m1_idx = m1Idx;
e.m2_plane = m2.plane_id;
e.m2_roi = m2.roi_idx;
e2 = merged(1);
fn = fieldnames(merged);
for i = 1:numel(fn)
    if isfield(e, fn{i})
        e2.(fn{i}) = e.(fn{i});
    end
end
merged(end+1) = e2; %#ok<AGROW>
save(dictPath, 'merged');
locPath = fullfile(cfg.outDir, 'localization.csv');
loc = read_csv(locPath);
n = numel(loc);
loc(n+1).idx = m1Idx;
loc(n+1).orig_idx = m1Idx;
loc(n+1).col = col;
loc(n+1).row = rowY;
loc(n+1).z_um = z;
loc(n+1).z_plane_equiv = z / 2;
loc(n+1).x_um = col * cfg.pixelUm;
loc(n+1).y_um = rowY * cfg.pixelUm;
loc(n+1).n_views_matched = 4;
loc(n+1).peak_corr = corr;
loc(n+1).center_spread_px = 0;
loc(n+1).mean_trace_corr = corr;
loc(n+1).min_trace_corr = corr;
loc(n+1).primary_view = pView;
write_csv(locPath, loc, fieldnames(loc));
matchRows = read_csv(matchCsv);
nm = numel(matchRows);
matchRows(nm+1).m1_idx = m1Idx;
matchRows(nm+1).m2_plane = m2.plane_id;
matchRows(nm+1).m2_roi = m2.roi_idx;
matchRows(nm+1).col1 = col;
matchRows(nm+1).row1 = rowY;
matchRows(nm+1).z1 = z;
matchRows(nm+1).x2 = m2.x;
matchRows(nm+1).y2 = m2.y;
matchRows(nm+1).z2 = m2.z_um;
matchRows(nm+1).x1_in_m2 = x1m;
matchRows(nm+1).y1_in_m2 = y1m;
matchRows(nm+1).d_xy_px = dXy;
matchRows(nm+1).d_z_um = dZ;
matchRows(nm+1).corr = corr;
matchRows(nm+1).cost = cfg.wXy * dXy + cfg.wZ * dZ + cfg.wR * (1 - corr);
write_csv(matchCsv, matchRows, fieldnames(matchRows));
end
