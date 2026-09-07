function dedupe_m2(cfg)
%DEDUPE_M2 Cross-plane duplicate removal (d_xy<=10, r>=0.6, keep higher SNR).
if nargin < 1, cfg = dataset_config(); end
covDir = fullfile(cfg.outDir, 'match', 'm2_3d_coverage');
dedupeDir = fullfile(covDir, 'dedupe');
ensure_dir(dedupeDir);
m2Pool = load_method2(cfg);
n = numel(m2Pool);
if n == 0
    fprintf('no method2 ROIs; skip dedupe\n');
    return
end
snrs = arrayfun(@(m) snr_trace(m.trace), m2Pool);
xy = zeros(n, 2);
plane = zeros(n, 1);
for i = 1:n
    xy(i, :) = [m2Pool(i).x, m2Pool(i).y];
    plane(i) = m2Pool(i).plane_id;
end
T = min(arrayfun(@(m) numel(m.trace), m2Pool));
st = 3;
tr = zeros(n, numel(1:st:T));
for i = 1:n
    raw = double(m2Pool(i).trace(1:T));
    tr(i, :) = raw(1:st:end);
end
parent = 1:n;
pairRows = struct('i', {}, 'j', {}, 'plane_i', {}, 'roi_i', {}, 'plane_j', {}, ...
    'roi_j', {}, 'd_xy_px', {}, 'corr', {}, 'snr_i', {}, 'snr_j', {}, 'keep', {});
np = 0;
for i = 1:n
    for j = i + 1:n
        if plane(i) == plane(j), continue; end
        dxy = hypot(xy(i, 1) - xy(j, 1), xy(i, 2) - xy(j, 2));
        if dxy > cfg.dedupeDXyMax, continue; end
        r = corrcoef_scalar(tr(i, :), tr(j, :));
        if ~isfinite(r) || r < cfg.dedupeRMin, continue; end
        parent = uf_union(parent, i, j);
        np = np + 1;
        pairRows(np).i = i - 1;
        pairRows(np).j = j - 1;
        pairRows(np).plane_i = plane(i);
        pairRows(np).roi_i = m2Pool(i).roi_idx;
        pairRows(np).plane_j = plane(j);
        pairRows(np).roi_j = m2Pool(j).roi_idx;
        pairRows(np).d_xy_px = dxy;
        pairRows(np).corr = r;
        pairRows(np).snr_i = snrs(i);
        pairRows(np).snr_j = snrs(j);
        if snrs(i) >= snrs(j)
            pairRows(np).keep = 'i';
        else
            pairRows(np).keep = 'j';
        end
    end
end
comps = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:n
    r = uf_find(parent, i);
    if isKey(comps, r)
        comps(r) = [comps(r), i];
    else
        comps(r) = i;
    end
end
roots = comps.keys;
kept = [];
compRows = struct('component', {}, 'plane', {}, 'roi', {}, 'snr', {}, 'kept', {}, ...
    'kept_plane', {}, 'kept_roi', {}, 'n_in_component', {});
nc = 0;
nMulti = 0;
for ri = 1:numel(roots)
    members = comps(roots{ri});
    if numel(members) == 1
        kept(end+1) = members; %#ok<AGROW>
        continue
    end
    nMulti = nMulti + 1;
    [~, ix] = max(snrs(members));
    best = members(ix);
    kept(end+1) = best; %#ok<AGROW>
    for m = members
        nc = nc + 1;
        compRows(nc).component = roots{ri};
        compRows(nc).plane = plane(m);
        compRows(nc).roi = m2Pool(m).roi_idx;
        compRows(nc).snr = snrs(m);
        compRows(nc).kept = double(m == best);
        compRows(nc).kept_plane = plane(best);
        compRows(nc).kept_roi = m2Pool(best).roi_idx;
        compRows(nc).n_in_component = numel(members);
    end
end
kept = sort(kept);
fprintf('dedupe: pairs=%d  components_multi=%d  kept=%d / %d  dropped=%d\n', ...
    numel(pairRows), nMulti, numel(kept), n, n - numel(kept));

write_csv(fullfile(dedupeDir, 'duplicate_pairs_snr.csv'), pairRows, ...
    {'i', 'j', 'plane_i', 'roi_i', 'plane_j', 'roi_j', 'd_xy_px', 'corr', 'snr_i', 'snr_j', 'keep'});
write_csv(fullfile(dedupeDir, 'dedupe_components.csv'), compRows, ...
    {'component', 'plane', 'roi', 'snr', 'kept', 'kept_plane', 'kept_roi', 'n_in_component'});
keptRows = struct('plane_id', {}, 'roi_idx', {}, 'x', {}, 'y', {}, 'z_um_plane', {}, 'snr', {});
for i = 1:numel(kept)
    p = m2Pool(kept(i));
    keptRows(i).plane_id = p.plane_id;
    keptRows(i).roi_idx = p.roi_idx;
    keptRows(i).x = p.x;
    keptRows(i).y = p.y;
    keptRows(i).z_um_plane = p.z_um;
    keptRows(i).snr = snrs(kept(i));
end
write_csv(fullfile(dedupeDir, 'kept_rois.csv'), keptRows, ...
    {'plane_id', 'roi_idx', 'x', 'y', 'z_um_plane', 'snr'});
write_json(fullfile(dedupeDir, 'dedupe_summary.json'), struct( ...
    'n_before', n, 'n_after', numel(kept), 'n_dropped', n - numel(kept), ...
    'n_pairs', numel(pairRows), 'gates', struct('D_XY_MAX', cfg.dedupeDXyMax, 'R_MIN', cfg.dedupeRMin)));

catPath = fullfile(covDir, 'm2_3d_catalog.csv');
oldCat = containers.Map('KeyType', 'char', 'ValueType', 'any');
if exist(catPath, 'file')
    copyfile(catPath, fullfile(dedupeDir, 'm2_3d_catalog_before_dedupe.csv'));
    oldRows = read_csv(catPath);
    for i = 1:numel(oldRows)
        oldCat(sprintf('%d_%d', oldRows(i).m2_plane, oldRows(i).m2_roi)) = oldRows(i);
    end
end
matchDir = resolve_match_dir(cfg);
matches = load_matches(fullfile(matchDir, 'matches.csv'));
[trustedAll, ~] = classify_matches(matches, cfg);
keptKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(kept)
    keptKeys(sprintf('%d_%d', m2Pool(kept(i)).plane_id, m2Pool(kept(i)).roi_idx)) = true;
end
trusted = trustedAll([]);
for i = 1:numel(trustedAll)
    k = sprintf('%d_%d', trustedAll(i).m2_plane, trustedAll(i).m2_roi);
    if isKey(keptKeys, k)
        if isempty(trusted)
            trusted = trustedAll(i);
        else
            trusted(end+1) = trustedAll(i); %#ok<AGROW>
        end
    end
end
[entriesM1, c1Orig, primaries] = load_method1(cfg);
catalogRows = [];
for ii = 1:numel(kept)
    p = m2Pool(kept(ii));
    key = sprintf('%d_%d', p.plane_id, p.roi_idx);
    row = struct('m2_plane', p.plane_id, 'm2_roi', p.roi_idx, 'source', '', 'col', '', ...
        'row', '', 'z_um', '', 'corr_or_best_r', '', 'primary_view', '', 'm1_idx', -1, ...
        'd_xy_px', '', 'd_z_um', '', 'n_pass', '', 'accepted', 1, 'snr', snrs(kept(ii)));
    hit = [];
    for t = 1:numel(trusted)
        if trusted(t).m2_plane == p.plane_id && trusted(t).m2_roi == p.roi_idx
            hit = trusted(t);
            break
        end
    end
    if ~isempty(hit)
        i1 = hit.m1_idx + 1;
        row.source = 'matched_trusted';
        row.col = c1Orig(i1, 1);
        row.row = c1Orig(i1, 2);
        row.z_um = c1Orig(i1, 3);
        row.corr_or_best_r = hit.corr;
        row.primary_view = primaries(i1);
        row.m1_idx = hit.m1_idx;
        row.d_xy_px = hit.d_xy_px;
        row.d_z_um = hit.d_z_um;
    elseif isKey(oldCat, key)
        old = oldCat(key);
        src = csv_field(old, 'source', '');
        zu = csv_field(old, 'z_um', []);
        acc = csv_field(old, 'accepted', 1);
        if (strcmp(src, 'recovered_bin') || strcmp(src, 'recovered_bin_refined')) ...
                && ~isempty(zu) && isnumeric(zu) && acc ~= 0
            row.source = 'recovered_bin_refined';
            row.col = old.col;
            row.row = old.row;
            row.z_um = old.z_um;
            row.corr_or_best_r = csv_field(old, 'corr_or_best_r', '');
            row.primary_view = csv_field(old, 'primary_view', '');
            row.d_z_um = abs(double(old.z_um) - plane_z_of(p.plane_id, cfg));
            row.n_pass = csv_field(old, 'n_pass', '');
        else
            row.source = 'failed';
            row.accepted = 0;
        end
    else
        row.source = 'failed';
        row.accepted = 0;
    end
    if isempty(catalogRows)
        catalogRows = row;
    else
        catalogRows(end+1) = row; %#ok<AGROW>
    end
end
[~, ord] = sortrows([[catalogRows.m2_plane]', [catalogRows.m2_roi]']);
catalogRows = catalogRows(ord);
write_csv(catPath, catalogRows, fieldnames(catalogRows));
bySrc = struct();
n3d = 0;
for i = 1:numel(catalogRows)
    s = catalogRows(i).source;
    if ~isfield(bySrc, s)
        bySrc.(s) = 1;
    else
        bySrc.(s) = bySrc.(s) + 1;
    end
    zu = catalogRows(i).z_um;
    if isnumeric(zu) && ~isempty(zu)
        n3d = n3d + 1;
    end
end
write_json(fullfile(covDir, 'coverage_summary.json'), struct( ...
    'n_m2_before_dedupe', n, 'n_m2_after_dedupe', numel(kept), 'n_with_3d', n3d, 'by_source', bySrc));
fprintf('coverage after dedupe: %d with 3D / %d kept\n', n3d, numel(kept));
fprintf('\nDone.\n');
end

function r = uf_find(parent, x)
while parent(x) ~= x
    parent(x) = parent(parent(x));
    x = parent(x);
end
r = x;
end

function parent = uf_union(parent, a, b)
ra = uf_find(parent, a);
rb = uf_find(parent, b);
if ra ~= rb
    parent(rb) = ra;
end
end
