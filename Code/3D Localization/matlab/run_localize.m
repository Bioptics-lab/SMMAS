function run_localize(cfg)
%RUN_LOCALIZE Method1 4-view localization (PSF calib, depth, merge, filter).
if nargin < 1, cfg = dataset_config(); end
ensure_dir(cfg.outDir);
fprintf('DATA_ROOT = %s\n', cfg.dataRoot);
fprintf('OUT_DIR   = %s\n', cfg.outDir);
fprintf('FACTOR=%.3f  MOVE_DIVISOR=%.3f  PSF_DZ_UM=%.2f  T_STRIDE=%d  DIST_THR=%.1f\n', ...
    cfg.factor, cfg.moveDivisor, cfg.psfDzUm, cfg.tStride, cfg.distThr);

if ~views_movies_available(cfg)
    fprintf(['No TIFF/data.bin for all views; skip Method1 localization.\n' ...
        'Restore raw movies under Views/{1-4}/ and re-run.\n']);
    return
end

fprintf('\n=== 1/4 PSF calibration ===\n');
coefPath = fullfile(cfg.outDir, 'psf_fit_coef.mat');
if exist(coefPath, 'file') && ~cfg.forceRecalib
    fprintf('reusing existing %s\n', coefPath);
    calib = load(coefPath);
else
    if cfg.forceRecalib
        fprintf('FORCE_RECALIB: ignoring existing %s\n', coefPath);
    end
    calib = calibrate_psf(cfg);
end
coefXz = calib.coef_psf_xz;
coefYz = calib.coef_psf_yz;
x0 = calib.x0;
y0 = calib.y0;
centers = calib.centers_1based;
zUm = calib.z_um;
halfSpan = max(abs(zUm(1)), abs(zUm(end)));
nSteps = ceil(halfSpan / cfg.depthStepUm);
depthMin = -nSteps * cfg.depthStepUm;
depthMax = nSteps * cfg.depthStepUm;
fprintf('depth search: [%d, %d] um step=%d\n', depthMin, depthMax, cfg.depthStepUm);

move = compute_move(centers, cfg.moveDivisor);
fprintf('move=\n');
disp(move);
fprintf('x0=%.2f y0=%.2f\n', x0, y0);

fprintf('\n=== 2/4 load suite2p ===\n');
s2p = cell(1, 4);
for v = 1:4
    s2p{v} = load_suite2p(view_plane_dir(v, cfg));
    fprintf('view%d: ROIs=%d iscell=%d\n', v, numel(s2p{v}.stat), numel(s2p{v}.Order_neuron));
end

fprintf('\n=== 3/4 load movies (block-average bin=%d) ===\n', cfg.tStride);
views = load_views_movies(cfg);
fprintf('views shape (Y,X,T,V)=%s\n', mat2str(size(views)));
[views, move, s2p] = apply_integer_move(views, move, s2p);
fprintf('applied integer move; estimate_depth move now zero\n');

fprintf('\n=== 4/4 localize + merge ===\n');
dicts = cell(1, 4);
for view = 1:4
    fprintf('estimate_depth_other primary view=%d (iscell n=%d) ...\n', ...
        view, numel(s2p{view}.Order_neuron));
    viewi = views(:, :, :, view);
    [neuronCenteri, cori, ~, rawtrace, pixels, centersAllview] = estimate_depth_other( ...
        s2p{view}.stat, s2p{view}.Order_neuron, coefXz, coefYz, viewi, views, view, move, ...
        cfg.factor, x0, y0, depthMin, depthMax, cfg.depthStepUm);
    d = build_dictionary(neuronCenteri, rawtrace, centersAllview, pixels, cori);
    peak = max(cori, [], 1, 'omitnan');
    for i = 1:numel(d)
        if i <= numel(peak)
            d(i).peak_corr = peak(i);
        else
            d(i).peak_corr = nan;
        end
        d(i).primary_view = view;
    end
    fprintf('  view%d neurons=%d\n', view, numel(d));
    dicts{view} = d;
end
nT = size(views, 3);
dicts = repair_traces_from_F(dicts, s2p, cfg.tStride, nT);

merged = merge_dictionary1(dicts{1}, dicts{2}, dicts{3}, dicts{4}, cfg.distThr, cfg.corrThr);
fprintf('merged neurons=%d (per-view: %s)\n', numel(merged), mat2str(cellfun(@numel, dicts)));
save(fullfile(cfg.outDir, 'neuron_dictionary_all.mat'), 'merged');

fprintf('\n=== quality filter (traces from block-averaged views) ===\n');
tracesForFilter = extract_traces_from_views(merged, views);
[kept, rejected, rejectInfo] = filter_merged_neurons(merged, tracesForFilter, cfg);
write_csv(fullfile(cfg.outDir, 'localization_rejected.csv'), rejectInfo, ...
    {'orig_idx', 'peak_corr', 'center_spread_px', 'mean_trace_corr', 'min_trace_corr', ...
    'best_center_views', 'best_corr_views', 'reasons'});
fprintf('wrote %s n=%d\n', fullfile(cfg.outDir, 'localization_rejected.csv'), numel(rejectInfo));

coords = export_localization(kept, cfg);
if ~isempty(coords)
    fprintf('\nkept depth um: mean=%.2f  std=%.2f  min=%.2f  max=%.2f\n', ...
        mean(coords(:, 3)), std(coords(:, 3)), min(coords(:, 3)), max(coords(:, 3)));
end
fprintf('\nDone. kept=%d rejected=%d  Outputs in %s\n', numel(kept), numel(rejected), cfg.outDir);
end

function [views, move, s2p] = apply_integer_move(views, move, s2p)
moveI = round(move);
for v = 1:size(views, 4)
    m0 = moveI(v, 1);
    m1 = moveI(v, 2);
    if m0 ~= 0 || m1 ~= 0
        views(:, :, :, v) = circshift(views(:, :, :, v), [-m1, -m0, 0]);
    end
    for i = 1:numel(s2p{v}.stat)
        s2p{v}.stat(i).xpix = double(s2p{v}.stat(i).xpix) - m1;
        s2p{v}.stat(i).ypix = double(s2p{v}.stat(i).ypix) - m0;
    end
end
move = zeros(size(move));
end

function dicts = repair_traces_from_F(dicts, s2p, tStride, nT)
for v = 1:4
    F = s2p{v}.F;
    order = s2p{v}.Order_neuron(:);
    for j = 1:numel(dicts{v})
        tr = double(dicts{v}(j).trace(:));
        if ~isempty(tr) && std(tr) > 1e-3
            continue
        end
        oi = order(j);
        if oi < 1 || oi > size(F, 1)
            continue
        end
        rawF = double(F(oi, :));
        nAvg = floor(numel(rawF) / tStride);
        if nAvg < 1
            continue
        end
        ftr = mean(reshape(rawF(1:nAvg * tStride), tStride, nAvg), 1);
        ftr = ftr(1:min(nT, numel(ftr)));
        if numel(ftr) < nT
            pad = zeros(1, nT);
            pad(1:numel(ftr)) = ftr;
            ftr = pad;
        end
        dicts{v}(j).trace = ftr;
        fprintf('  repaired trace view%d neuron%d from F.npy\n', v, j);
    end
end
end

function coords = export_localization(merged, cfg)
outDir = cfg.outDir;
ensure_dir(outDir);
save(fullfile(outDir, 'neuron_dictionary.mat'), 'merged');
n = numel(merged);
coords = zeros(n, 3);
rows = struct('idx', {}, 'orig_idx', {}, 'col', {}, 'row', {}, 'z_um', {}, ...
    'z_plane_equiv', {}, 'x_um', {}, 'y_um', {}, 'n_views_matched', {}, ...
    'peak_corr', {}, 'center_spread_px', {}, 'mean_trace_corr', {}, ...
    'min_trace_corr', {}, 'primary_view', {});
for i = 1:n
    c = double(merged(i).center(:))';
    col = c(1); rowY = c(2); zUm = c(3);
    coords(i, :) = [col, rowY, zUm];
    peak = nan;
    if isfield(merged(i), 'peak_corr')
        peak = double(merged(i).peak_corr);
    end
    orig = i - 1;
    if isfield(merged(i), 'orig_idx')
        orig = merged(i).orig_idx;
    end
    spread = nan; meanR = nan; minR = nan; pv = -1;
    if isfield(merged(i), 'center_spread_px'), spread = merged(i).center_spread_px; end
    if isfield(merged(i), 'mean_trace_corr'), meanR = merged(i).mean_trace_corr; end
    if isfield(merged(i), 'min_trace_corr'), minR = merged(i).min_trace_corr; end
    if isfield(merged(i), 'primary_view'), pv = merged(i).primary_view; end
    rows(i).idx = i - 1;
    rows(i).orig_idx = orig;
    rows(i).col = col;
    rows(i).row = rowY;
    rows(i).z_um = zUm;
    rows(i).z_plane_equiv = zUm / cfg.psfDzUm;
    rows(i).x_um = col * cfg.pixelUm;
    rows(i).y_um = rowY * cfg.pixelUm;
    rows(i).n_views_matched = n_views_matched(merged(i));
    rows(i).peak_corr = peak;
    rows(i).center_spread_px = spread;
    rows(i).mean_trace_corr = meanR;
    rows(i).min_trace_corr = minR;
    rows(i).primary_view = pv;
end
csvPath = fullfile(outDir, 'localization.csv');
write_csv(csvPath, rows, {'idx', 'orig_idx', 'col', 'row', 'z_um', 'z_plane_equiv', ...
    'x_um', 'y_um', 'n_views_matched', 'peak_corr', 'center_spread_px', ...
    'mean_trace_corr', 'min_trace_corr', 'primary_view'});
fprintf('wrote %s\n', csvPath);
if n == 0
    return
end
fig = figure('Visible', 'off');
scatter(coords(:, 1), coords(:, 2), 28, coords(:, 3), 'filled');
axis equal; set(gca, 'YDir', 'reverse');
xlabel('col (px)'); ylabel('row (px)');
title('TPLFM LinearRunner XY (color = z um)');
colorbar;
print(fig, fullfile(outDir, 'localization_xy.png'), '-dpng', '-r120');
close(fig);
fig = figure('Visible', 'off');
scatter3(coords(:, 1), coords(:, 2), coords(:, 3), 28, coords(:, 3), 'filled');
xlabel('col (px)'); ylabel('row (px)'); zlabel('z (um)');
title('TPLFM LinearRunner 3D localization');
print(fig, fullfile(outDir, 'localization_3d.png'), '-dpng', '-r120');
close(fig);
fig = figure('Visible', 'off');
zFinite = coords(:, 3);
zFinite = zFinite(isfinite(zFinite));
if ~isempty(zFinite)
    histogram(zFinite, max(8, min(40, floor(numel(zFinite) / 2))));
end
xlabel('z (um)'); ylabel('count');
title(sprintf('Depth histogram (n=%d finite)', numel(zFinite)));
print(fig, fullfile(outDir, 'depth_hist.png'), '-dpng', '-r120');
close(fig);
peakVals = [rows.peak_corr];
peakFinite = peakVals(isfinite(peakVals));
fig = figure('Visible', 'off');
if ~isempty(peakFinite)
    histogram(peakFinite, max(10, min(40, floor(numel(peakFinite) / 5))));
    hold on;
    xline(median(peakFinite), '--');
    xline(mean(peakFinite), ':');
end
xlabel('peak_corr'); ylabel('count');
title(sprintf('peak_corr distribution (n=%d)', numel(peakFinite)));
print(fig, fullfile(outDir, 'peak_corr_hist.png'), '-dpng', '-r140');
close(fig);
fprintf('wrote localization plots\n');
end

function n = n_views_matched(entry)
cav = reshape(double(entry.center_allview), 2, 4);
n = sum(any(abs(cav) > 1e-6, 1));
if n < 1, n = 1; end
end
