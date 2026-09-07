function [kept, rejected, rejectInfo] = filter_merged_neurons(merged, tracesN4t, cfg)
%FILTER_MERGED_NEURONS Keep neurons with a usable best-k-of-4 subset.
if nargin < 3, cfg = dataset_config(); end
if ~isempty(merged)
    merged(1).center_spread_px = nan;
    merged(1).mean_trace_corr = nan;
    merged(1).min_trace_corr = nan;
    merged(1).best_center_views = '';
    merged(1).best_corr_views = '';
    merged(1).orig_idx = 0;
end
minPeak = cfg.filterMinPeakCorr;
maxSpread = cfg.filterMaxCenterSpreadPx;
minMeanR = cfg.filterMinMeanTraceCorr;
minViews = cfg.filterMinViews;
kept = merged([]);
rejected = merged([]);
rejectInfo = struct('orig_idx', {}, 'peak_corr', {}, 'center_spread_px', {}, ...
    'mean_trace_corr', {}, 'min_trace_corr', {}, 'best_center_views', {}, ...
    'best_corr_views', {}, 'reasons', {});
for i = 1:numel(merged)
    entry = merged(i);
    peak = nan;
    if isfield(entry, 'peak_corr')
        peak = double(entry.peak_corr);
    end
    if ~isfinite(peak) && isfield(entry, 'cori_allneuron_allz')
        curve = double(entry.cori_allneuron_allz(:));
        if ~isempty(curve)
            peak = max(curve(isfinite(curve)), [], 'omitnan');
        end
    end
    [spread, spreadViews] = best_k_center_spread(entry, minViews);
    meanR = nan;
    corrViews = [];
    if ~isempty(tracesN4t)
        [meanR, corrViews] = best_k_trace_corr(squeeze(tracesN4t(i, :, :)), minViews);
    end
    nFinite = sum(~cellfun(@isempty, view_centers_xy(entry)));
    reasons = {};
    if nFinite < minViews
        reasons{end+1} = sprintf('n_views=%d<%d', nFinite, minViews); %#ok<AGROW>
    end
    if ~isfinite(peak) || peak < minPeak
        reasons{end+1} = sprintf('peak_corr=%.3f<%.3f', peak, minPeak); %#ok<AGROW>
    end
    if ~isfinite(spread) || spread > maxSpread
        reasons{end+1} = sprintf('best%d_spread=%.1f>%.1f', minViews, spread, maxSpread); %#ok<AGROW>
    end
    if ~isempty(tracesN4t) && (~isfinite(meanR) || meanR < minMeanR)
        reasons{end+1} = sprintf('best%d_mean_r=%.3f<%.3f', minViews, meanR, minMeanR); %#ok<AGROW>
    end
    entry.peak_corr = peak;
    entry.center_spread_px = spread;
    entry.mean_trace_corr = meanR;
    entry.min_trace_corr = meanR;
    entry.best_center_views = join_views(spreadViews);
    entry.best_corr_views = join_views(corrViews);
    entry.orig_idx = i - 1;
    meta = struct( ...
        'orig_idx', i - 1, ...
        'peak_corr', peak, ...
        'center_spread_px', spread, ...
        'mean_trace_corr', meanR, ...
        'min_trace_corr', meanR, ...
        'best_center_views', entry.best_center_views, ...
        'best_corr_views', entry.best_corr_views, ...
        'reasons', strjoin(reasons, '; '));
    if isempty(reasons)
        kept = [kept; entry]; %#ok<AGROW>
    else
        rejected = [rejected; entry]; %#ok<AGROW>
        rejectInfo = [rejectInfo; meta]; %#ok<AGROW>
    end
end
fprintf('filter (best-%d-of-4): keep %d / reject %d  (peak>=%.2f, spread<=%.1fpx, mean_r>=%.2f)\n', ...
    minViews, numel(kept), numel(rejected), minPeak, maxSpread, minMeanR);
if ~isempty(rejectInfo)
    nShow = min(8, numel(rejectInfo));
    for i = 1:nShow
        fprintf('  reject orig#%d: %s\n', rejectInfo(i).orig_idx, rejectInfo(i).reasons);
    end
    if numel(rejectInfo) > 8
        fprintf('  ... (%d more)\n', numel(rejectInfo) - 8);
    end
end
end

function pts = view_centers_xy(entry)
cav = reshape(double(entry.center_allview), 2, 4);
pts = cell(1, 4);
for v = 1:4
    x = cav(1, v); y = cav(2, v);
    if isfinite(x) && isfinite(y) && (abs(x) > 1e-6 || abs(y) > 1e-6)
        pts{v} = [x, y];
    else
        pts{v} = [];
    end
end
end

function [bestSpread, bestSet] = best_k_center_spread(entry, k)
centers = view_centers_xy(entry);
idxs = find(~cellfun(@isempty, centers));
if numel(idxs) < k
    bestSpread = inf;
    bestSet = [];
    return
end
combs = nchoosek(idxs, k);
bestSpread = inf;
bestSet = [];
for i = 1:size(combs, 1)
    pts = zeros(k, 2);
    for j = 1:k
        pts(j, :) = centers{combs(i, j)};
    end
    sp = max_pairwise(pts);
    if sp < bestSpread
        bestSpread = sp;
        bestSet = combs(i, :);
    end
end
end

function dmax = max_pairwise(pts)
dmax = 0;
for i = 1:size(pts, 1)
    for j = i + 1:size(pts, 1)
        d = hypot(pts(i, 1) - pts(j, 1), pts(i, 2) - pts(j, 2));
        dmax = max(dmax, d);
    end
end
end

function [bestMean, bestSet] = best_k_trace_corr(traces, k)
if size(traces, 1) ~= 4
    traces = reshape(traces, 4, []);
end
usable = [];
for i = 1:4
    if std(double(traces(i, :))) > 1e-6
        usable(end+1) = i; %#ok<AGROW>
    end
end
if numel(usable) < k
    bestMean = nan;
    bestSet = [];
    return
end
combs = nchoosek(usable, k);
bestMean = -inf;
bestSet = [];
for i = 1:size(combs, 1)
    comb = combs(i, :);
    rs = [];
    pairs = nchoosek(comb, 2);
    for p = 1:size(pairs, 1)
        r = corrcoef_scalar(traces(pairs(p, 1), :), traces(pairs(p, 2), :));
        if isfinite(r)
            rs(end+1) = r; %#ok<AGROW>
        end
    end
    if isempty(rs), continue; end
    m = mean(rs);
    if m > bestMean
        bestMean = m;
        bestSet = comb;
    end
end
if isempty(bestSet)
    bestMean = nan;
end
end

function s = join_views(vs)
if isempty(vs)
    s = '';
else
    s = strjoin(arrayfun(@(x) num2str(x), vs, 'UniformOutput', false), ',');
end
end
