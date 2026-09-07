function refine_z_from_footprints(cfg)
%REFINE_Z_FROM_FOOTPRINTS Refine recovered m2 3D on block-averaged movies + A/B z-sign.
if nargin < 1, cfg = dataset_config(); end
covDir = fullfile(cfg.outDir, 'match', 'm2_3d_coverage');
refineDir = fullfile(covDir, 'z_refined');
ensure_dir(refineDir);
catPath = fullfile(covDir, 'm2_3d_catalog.csv');
if ~exist(catPath, 'file')
    fprintf('missing %s; skip refine\n', catPath);
    return
end
rows = read_csv(catPath);
matchDir = resolve_match_dir(cfg);
matches = load_matches(fullfile(matchDir, 'matches.csv'));
[trusted, ~] = classify_matches(matches, cfg);
trustedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(trusted)
    trustedKeys(sprintf('%d_%d', trusted(i).m2_plane, trusted(i).m2_roi)) = true;
end
toRefine = [];
for i = 1:numel(rows)
    src = csv_field(rows(i), 'source', '');
    key = sprintf('%d_%d', rows(i).m2_plane, rows(i).m2_roi);
    z = csv_field(rows(i), 'z_um', []);
    if (strcmp(src, 'recovered_bin') || strcmp(src, 'recovered_bin_refined')) ...
            && ~isKey(trustedKeys, key) && ~isempty(z) && isnumeric(z)
        if isempty(toRefine)
            toRefine = rows(i);
        else
            toRefine(end+1) = rows(i); %#ok<AGROW>
        end
    end
end
fprintf('refine candidates: %d  trusted skipped: %d  AVG_BIN=%d\n', ...
    numel(toRefine), trustedKeys.Count, cfg.avgBin);
if isempty(toRefine)
    fprintf('nothing to refine; catalog left as recover wrote it\n');
    write_json(fullfile(refineDir, 'refine_summary.json'), struct('n_refined', 0, 'trusted_skipped', trustedKeys.Count));
    return
end
if ~views_movies_available(cfg)
    fprintf('no TIFF/data.bin; skip refine\n');
    return
end

[~, c1Orig, ~] = load_method1(cfg);
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
[mems, ~, ~, ~] = open_view_memmaps(cfg);
fprintf('\n=== block-average volumes ===\n');
vols = average_view_volumes(mems, cfg.avgBin);
fprintf('\n=== refine on averaged volumes ===\n');
results = [];
for k = 1:numel(toRefine)
    row = toRefine(k);
    key = sprintf('%d_%d', row.m2_plane, row.m2_roi);
    m2 = m2Pool(m2Map(key));
    res = refine_one(m2, row.col, row.row, row.z_um, tfm, vols, moveI, coefXz, coefYz, factor, x0, y0, cfg);
    if isempty(results)
        results = res;
    else
        results(end+1) = res; %#ok<AGROW>
    end
    fprintf('  [%d/%d] p%d/roi%d  z %.1f -> %.1f (%s)  r=%.3f\n', ...
        k, numel(toRefine), row.m2_plane, row.m2_roi, res.z_old, res.best_z, res.z_source, res.best_r);
end

fprintf('\n=== A/B plane z sign (after all depths) ===\n');
ab = ab_test_from_depths(results, trusted, c1Orig, cfg);
fprintf('choose %s\n', ab.choose);
write_json(fullfile(refineDir, 'plane_z_ab_test.json'), ab);
set_plane_z_flipped(strcmp(ab.choose, 'B'), cfg);

fields = {'m2_plane', 'm2_roi', 'col', 'row', 'z_old', 'best_z', 'z_activity', ...
    'z_parallax', 'z_source', 'parallax_rms_px', 'best_r', 'n_pass', 'primary_view', ...
    'r_v1', 'r_v2', 'r_v3', 'r_v4', 'plane_z_prior'};
rep = struct([]);
for i = 1:numel(results)
    r = results(i);
    pid = r.m2.plane_id;
    rep(i).m2_plane = pid;
    rep(i).m2_roi = r.m2.roi_idx;
    rep(i).col = r.col;
    rep(i).row = r.row;
    rep(i).z_old = r.z_old;
    rep(i).best_z = r.best_z;
    rep(i).z_activity = r.z_activity;
    rep(i).z_parallax = r.z_parallax;
    rep(i).z_source = r.z_source;
    rep(i).parallax_rms_px = r.parallax_rms_px;
    rep(i).best_r = r.best_r;
    rep(i).n_pass = r.n_pass;
    rep(i).primary_view = r.primary_v0 + 1;
    rs = r.rs;
    rep(i).r_v1 = rs(1); rep(i).r_v2 = rs(2); rep(i).r_v3 = rs(3); rep(i).r_v4 = rs(4);
    rep(i).plane_z_prior = plane_z_of(pid, cfg);
end
write_csv(fullfile(refineDir, 'refine_report.csv'), rep, fields);

refineMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(results)
    refineMap(sprintf('%d_%d', results(i).m2.plane_id, results(i).m2.roi_idx)) = i;
end
newRows = rows;
for i = 1:numel(newRows)
    key = sprintf('%d_%d', newRows(i).m2_plane, newRows(i).m2_roi);
    if isKey(refineMap, key)
        r = results(refineMap(key));
        newRows(i).col = r.col;
        newRows(i).row = r.row;
        newRows(i).z_um = r.best_z;
        newRows(i).corr_or_best_r = r.best_r;
        newRows(i).primary_view = r.primary_v0 + 1;
        newRows(i).source = 'recovered_bin_refined';
        newRows(i).d_z_um = abs(r.best_z - plane_z_of(r.m2.plane_id, cfg));
        newRows(i).n_pass = r.n_pass;
    end
end
write_csv(catPath, newRows, fieldnames(newRows));

dictPath = fullfile(covDir, 'm2_3d_dictionary.mat');
kept = [];
if exist(dictPath, 'file')
    d = load(dictPath);
    old = d.dictOut;
    if iscell(old)
        oldList = old;
    else
        oldList = num2cell(old);
    end
    for i = 1:numel(oldList)
        e = oldList{i};
        if isfield(e, 'source') && strcmp(e.source, 'matched_trusted')
            if isempty(kept)
                kept = e;
            else
                kept(end+1) = e; %#ok<AGROW>
            end
        end
    end
end
recoveredEntries = [];
for i = 1:numel(results)
    e = build_synthetic_entry(results(i), coefXz, coefYz, factor, x0, y0);
    e.source = 'recovered_bin_refined';
    e.z_source = results(i).z_source;
    e.z_old = results(i).z_old;
    if isempty(recoveredEntries)
        recoveredEntries = e;
    else
        recoveredEntries(end+1) = e; %#ok<AGROW>
    end
end
if isempty(kept)
    dictOut = num2cell(recoveredEntries);
elseif isempty(recoveredEntries)
    dictOut = num2cell(kept);
else
    dictOut = [num2cell(kept(:)); num2cell(recoveredEntries(:))];
end
if ~isempty(dictOut)
    save(dictPath, 'dictOut');
end
zNew = [results.best_z];
summary = struct('n_refined', numel(results), 'avg_bin', cfg.avgBin, ...
    'plane_z_convention', ab.choose, 'median_z', median(zNew), ...
    'z_min', min(zNew), 'z_max', max(zNew), 'ab_test', ab);
write_json(fullfile(refineDir, 'refine_summary.json'), summary);
fprintf('summary n_refined=%d choose=%s\n', numel(results), ab.choose);
fprintf('\nDone.\n');
end
