function match_to_recon_planes(cfg)
%MATCH_TO_RECON_PLANES Hungarian match of method1 neurons to recon-plane ROIs.
if nargin < 1, cfg = dataset_config(); end
mode = cfg.matchMode;
gates = cfg.matchGates.(mode);
outDir = fullfile(cfg.outDir, 'match', mode);
ensure_dir(outDir);
fprintf('OUT_DIR = %s\n', outDir);
fprintf('MATCH_MODE=%s  D_XY=%.1fpx  D_Z=%.1fum  R_MIN=%.2f  W=(%.1f,%.1f,%.1f)\n', ...
    mode, gates.dXyPx, gates.dZUm, gates.rMin, cfg.wXy, cfg.wZ, cfg.wR);

if ~exist(fullfile(cfg.outDir, 'localization.csv'), 'file')
    fprintf('missing localization.csv; skip match\n');
    return
end

fprintf('\n=== load method1 / method2 ===\n');
[entries, c1Orig, primaries] = load_method1(cfg);
m2Pool = load_method2(cfg);

fprintf('\n=== XY registration ===\n');
img1 = load_m1_mean_image(cfg);
img2 = load_m2_ref_mean_image(cfg);
tfm = estimate_xy_transform(img1, img2);
save(fullfile(outDir, 'xy_transform.mat'), '-struct', 'tfm');
plot_xy_overlay(img1, img2, tfm, fullfile(outDir, 'xy_overlay.png'), cfg.matchMidPlane);

[x1m2, y1m2] = apply_xy(c1Orig(:, 1), c1Orig(:, 2), tfm);
c1M2 = [x1m2, y1m2, c1Orig(:, 3)];

fprintf('\n=== method1 traces ===\n');
t1 = extract_m1_primary_traces(entries, primaries, cfg);
t2 = zeros(numel(m2Pool), numel(m2Pool(1).trace));
for j = 1:numel(m2Pool)
    t2(j, :) = m2Pool(j).trace;
end
nT = min(size(t1, 2), size(t2, 2));
t1 = t1(:, 1:nT);
t2 = t2(:, 1:nT);
step = max(cfg.traceTStride, 1);
if step > 1
    t1 = t1(:, 1:step:end);
    t2 = t2(:, 1:step:end);
end
fprintf('traces T=%d  stride=%d  m1=%s  m2=%s\n', size(t1, 2), step, mat2str(size(t1)), mat2str(size(t2)));

fprintf('\n=== Hungarian match ===\n');
x2 = arrayfun(@(m) m.x, m2Pool);
y2 = arrayfun(@(m) m.y, m2Pool);
z2 = arrayfun(@(m) m.z_um, m2Pool);
[rInd, cInd, dxy, dz, corr] = build_cost_and_match(c1M2, x2, y2, z2, t1, t2, gates, cfg);
costVals = cfg.wXy * dxy + cfg.wZ * dz + cfg.wR * (1.0 - corr);
planeHits = containers.Map('KeyType', 'double', 'ValueType', 'double');
for k = 1:numel(cInd)
    pid = m2Pool(cInd(k)).plane_id;
    if isKey(planeHits, pid)
        planeHits(pid) = planeHits(pid) + 1;
    else
        planeHits(pid) = 1;
    end
end
fprintf('matched %d / m1=%d / m2=%d\n', numel(rInd), size(c1Orig, 1), numel(m2Pool));
if ~isempty(corr)
    fprintf('median d_xy=%.2f px  median |dz|=%.2f um  median r=%.3f\n', ...
        median(dxy), median(dz), median(corr));
end
ks = planeHits.keys;
for i = 1:numel(ks)
    fprintf('  plane %d hits=%d\n', ks{i}, planeHits(ks{i}));
end
export_match_results(outDir, c1Orig, c1M2, m2Pool, rInd, cInd, dxy, dz, corr, costVals);
fprintf('\nDone.\n');
end

function img = load_m1_mean_image(cfg)
imgs = [];
for v = 1:4
    s2p = load_suite2p(view_plane_dir(v, cfg));
    im = s2p.ops.meanImg;
    if isempty(imgs)
        imgs = im;
    else
        imgs = max(imgs, im);
    end
end
img = imgs;
end

function img = load_m2_ref_mean_image(cfg)
s2p = load_suite2p(plane_suite2p_dir(cfg.matchMidPlane, cfg));
img = s2p.ops.meanImg;
end

function plot_xy_overlay(img1, img2, tfm, outPath, midPlane)
h2 = size(img2, 1); w2 = size(img2, 2);
[xx, yy] = meshgrid(0:w2-1, 0:h2-1);
srcX = (xx - tfm.tx) / tfm.s;
srcY = (yy - tfm.ty) / tfm.s;
warped = interp2(0:size(img1, 2)-1, 0:size(img1, 1)-1, double(img1), srcX, srcY, 'linear', 0);
fig = figure('Visible', 'off');
subplot(1, 3, 1); imagesc(img1); axis image off; colormap(gca, gray); title('method1 mean (Views)');
subplot(1, 3, 2); imagesc(img2); axis image off; colormap(gca, gray); title(sprintf('method2 Plane%d mean', midPlane));
a = warped - min(warped(:)); a = a / (max(a(:)) + 1e-12);
b = double(img2); b = b - min(b(:)); b = b / (max(b(:)) + 1e-12);
rgb = cat(3, a, b, 0.3 * a + 0.3 * b);
subplot(1, 3, 3); imagesc(min(max(rgb, 0), 1)); axis image off;
title(sprintf('overlay R=m1warp G=m2  NCC~%.3f', tfm.ncc));
print(fig, outPath, '-dpng', '-r120');
close(fig);
fprintf('wrote %s\n', outPath);
end

function t1 = extract_m1_primary_traces(entries, primaries, cfg)
n = numel(entries);
useBin = true;
try
    [mems, ly, lx, nf] = open_view_memmaps(cfg);
catch
    useBin = false;
    mems = [];
    ly = []; lx = []; nf = [];
end
if ~useBin
    fprintf('no data.bin; using dictionary traces for match\n');
    tlen = 0;
    for i = 1:n
        tlen = max(tlen, numel(entries(i).trace));
    end
    t1 = zeros(n, max(tlen, 1));
    for i = 1:n
        tr = double(entries(i).trace(:))';
        t1(i, 1:numel(tr)) = tr;
    end
    return
end
calib = load_psf_coef(cfg);
moveI = round(compute_move(calib.centers_1based, cfg.moveDivisor));
t1 = zeros(n, nf);
byView = cell(1, 4);
for i = 1:n
    pv = max(1, min(4, round(primaries(i))));
    byView{pv}(end+1) = i; %#ok<AGROW>
end
for pv = 1:4
    idxs = byView{pv};
    if isempty(idxs), continue; end
    v0 = pv;
    m0 = moveI(v0, 1); m1 = moveI(v0, 2);
    pix = cell(numel(idxs), 1);
    for k = 1:numel(idxs)
        e = entries(idxs(k));
        cav = reshape(double(e.center_allview), 2, 4);
        d1 = double(e.pixels1(:));
        d2 = double(e.pixels2(:));
        cx = cav(1, v0); cy = cav(2, v0);
        px = round(d1 + cx);
        py = round(d2 + cy);
        ox = min(max(px + m1, 0), lx - 1);
        oy = min(max(py + m0, 0), ly - 1);
        pix{k} = [oy, ox];
    end
    fprintf('  extract m1 primary view%d: %d neurons, T=%d\n', pv, numel(idxs), nf);
    chunk = 100;
    for s = 1:chunk:nf
        e = min(s + chunk - 1, nf);
        block = zeros(e - s + 1, ly, lx, 'single');
        for t = s:e
            block(t - s + 1, :, :) = single(read_bin_frame(mems{v0}, t));
        end
        for k = 1:numel(idxs)
            oy = pix{k}(:, 1); ox = pix{k}(:, 2);
            if isempty(oy), continue; end
            acc = zeros(e - s + 1, 1);
            for p = 1:numel(oy)
                acc = acc + double(block(:, oy(p) + 1, ox(p) + 1));
            end
            t1(idxs(k), s:e) = acc / numel(oy);
        end
    end
end
end

function [rInd, cInd, dxyOut, dzOut, rOut] = build_cost_and_match(c1, x2, y2, z2, t1, t2, gates, cfg)
n1 = size(c1, 1);
n2 = numel(x2);
BIG = 1e6;
cost = BIG * ones(n1, n2);
dxyM = nan(n1, n2);
dzM = nan(n1, n2);
rM = nan(n1, n2);
t1z = zeros(size(t1));
t2z = zeros(size(t2));
for i = 1:n1
    t1z(i, :) = zscore_vec(t1(i, :));
end
for j = 1:n2
    t2z(j, :) = zscore_vec(t2(j, :));
end
for i = 1:n1
    x1i = c1(i, 1); y1i = c1(i, 2); z1i = c1(i, 3);
    for j = 1:n2
        dxy = hypot(x1i - x2(j), y1i - y2(j));
        dz = abs(z1i - z2(j));
        if dxy > gates.dXyPx || dz > gates.dZUm
            continue
        end
        r = corrcoef_scalar(t1z(i, :), t2z(j, :));
        if ~isfinite(r) || r < gates.rMin
            continue
        end
        dxyM(i, j) = dxy;
        dzM(i, j) = dz;
        rM(i, j) = r;
        cost(i, j) = cfg.wXy * dxy + cfg.wZ * dz + cfg.wR * (1.0 - r);
    end
end
unmatched = BIG * 0.5;
if n1 == 0 || n2 == 0
    rInd = []; cInd = []; dxyOut = []; dzOut = []; rOut = [];
    return
end
M = matchpairs(cost, unmatched);
if isempty(M)
    rInd = []; cInd = []; dxyOut = []; dzOut = []; rOut = [];
    return
end
keep = false(size(M, 1), 1);
for k = 1:size(M, 1)
    keep(k) = cost(M(k, 1), M(k, 2)) < unmatched;
end
M = M(keep, :);
rInd = M(:, 1);
cInd = M(:, 2);
lin = sub2ind(size(cost), rInd, cInd);
dxyOut = dxyM(lin);
dzOut = dzM(lin);
rOut = rM(lin);
end

function export_match_results(outDir, c1Orig, c1M2, m2Pool, rInd, cInd, dxy, dz, corr, costVals)
nMatch = numel(rInd);
matchRows = struct('m1_idx', {}, 'm2_plane', {}, 'm2_roi', {}, 'col1', {}, 'row1', {}, ...
    'z1', {}, 'x2', {}, 'y2', {}, 'z2', {}, 'x1_in_m2', {}, 'y1_in_m2', {}, ...
    'd_xy_px', {}, 'd_z_um', {}, 'corr', {}, 'cost', {});
matchedM1 = false(size(c1Orig, 1), 1);
matchedM2 = false(numel(m2Pool), 1);
for k = 1:nMatch
    i = rInd(k); j = cInd(k);
    m2 = m2Pool(j);
    matchRows(k).m1_idx = i - 1;
    matchRows(k).m2_plane = m2.plane_id;
    matchRows(k).m2_roi = m2.roi_idx;
    matchRows(k).col1 = c1Orig(i, 1);
    matchRows(k).row1 = c1Orig(i, 2);
    matchRows(k).z1 = c1Orig(i, 3);
    matchRows(k).x2 = m2.x;
    matchRows(k).y2 = m2.y;
    matchRows(k).z2 = m2.z_um;
    matchRows(k).x1_in_m2 = c1M2(i, 1);
    matchRows(k).y1_in_m2 = c1M2(i, 2);
    matchRows(k).d_xy_px = dxy(k);
    matchRows(k).d_z_um = dz(k);
    matchRows(k).corr = corr(k);
    matchRows(k).cost = costVals(k);
    matchedM1(i) = true;
    matchedM2(j) = true;
end
fields = {'m1_idx', 'm2_plane', 'm2_roi', 'col1', 'row1', 'z1', 'x2', 'y2', 'z2', ...
    'x1_in_m2', 'y1_in_m2', 'd_xy_px', 'd_z_um', 'corr', 'cost'};
write_csv(fullfile(outDir, 'matches.csv'), matchRows, fields);
fprintf('wrote %s n=%d\n', fullfile(outDir, 'matches.csv'), nMatch);

um1 = struct('m1_idx', {}, 'col1', {}, 'row1', {}, 'z1', {});
k = 0;
for i = 1:size(c1Orig, 1)
    if ~matchedM1(i)
        k = k + 1;
        um1(k).m1_idx = i - 1;
        um1(k).col1 = c1Orig(i, 1);
        um1(k).row1 = c1Orig(i, 2);
        um1(k).z1 = c1Orig(i, 3);
    end
end
write_csv(fullfile(outDir, 'unmatched_m1.csv'), um1, {'m1_idx', 'col1', 'row1', 'z1'});
um2 = struct('m2_plane', {}, 'm2_roi', {}, 'x2', {}, 'y2', {}, 'z2', {});
k = 0;
for j = 1:numel(m2Pool)
    if ~matchedM2(j)
        k = k + 1;
        um2(k).m2_plane = m2Pool(j).plane_id;
        um2(k).m2_roi = m2Pool(j).roi_idx;
        um2(k).x2 = m2Pool(j).x;
        um2(k).y2 = m2Pool(j).y;
        um2(k).z2 = m2Pool(j).z_um;
    end
end
write_csv(fullfile(outDir, 'unmatched_m2.csv'), um2, {'m2_plane', 'm2_roi', 'x2', 'y2', 'z2'});
fprintf('unmatched m1=%d  m2=%d\n', numel(um1), numel(um2));

fig = figure('Visible', 'off');
scatter(arrayfun(@(m) m.x, m2Pool), arrayfun(@(m) m.y, m2Pool), 18, 0.7 * [1 1 1], 'filled');
hold on;
scatter(c1M2(:, 1), c1M2(:, 2), 28, c1Orig(:, 3), 'x');
for k = 1:nMatch
    j = cInd(k); i = rInd(k);
    plot([c1M2(i, 1), m2Pool(j).x], [c1M2(i, 2), m2Pool(j).y], '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 0.7);
end
axis equal; set(gca, 'YDir', 'reverse');
xlabel('x (method2 px)'); ylabel('y (method2 px)');
title(sprintf('matches n=%d', nMatch));
print(fig, fullfile(outDir, 'match_xy_scatter.png'), '-dpng', '-r120');
close(fig);
fig = figure('Visible', 'off');
if ~isempty(corr)
    histogram(corr, max(8, min(30, floor(numel(corr) / 3))));
    hold on; xline(median(corr), '--');
end
xlabel('trace corr'); ylabel('count'); title('match correlation');
print(fig, fullfile(outDir, 'match_corr_hist.png'), '-dpng', '-r120');
close(fig);
fprintf('wrote match_xy_scatter.png, match_corr_hist.png\n');
end
