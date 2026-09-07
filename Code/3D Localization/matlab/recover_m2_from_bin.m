function recover_m2_from_bin(cfg)
%RECOVER_M2_FROM_BIN Fill 3D for weak/unmatched plane ROIs via data.bin z-sweep.
if nargin < 1, cfg = dataset_config(); end
matchDir = resolve_match_dir(cfg);
covDir = fullfile(cfg.outDir, 'match', 'm2_3d_coverage');
recDir = fullfile(covDir, 'recovered_from_bin');
ensure_dir(covDir);
ensure_dir(recDir);
matchCsv = fullfile(matchDir, 'matches.csv');
if ~exist(matchCsv, 'file')
    fprintf('no matches.csv under %s; skip recover\n', matchDir);
    return
end
if ~views_movies_available(cfg)
    fprintf('no TIFF/data.bin; skip recover (trusted matches still catalogued if present)\n');
    recover_trusted_only(cfg, matchDir, covDir);
    return
end

fprintf('MATCH_DIR = %s\n', matchDir);
matches = load_matches(matchCsv);
[trusted, weak] = classify_matches(matches, cfg);
um2Path = fullfile(matchDir, 'unmatched_m2.csv');
if exist(um2Path, 'file')
    um2Rows = read_csv(um2Path);
else
    um2Rows = struct([]);
end
fprintf('matches=%d trusted=%d weak=%d unmatched_m2=%d\n', ...
    numel(matches), numel(trusted), numel(weak), numel(um2Rows));

fprintf('\n=== load method1 / method2 ===\n');
[entriesM1, c1Orig, primaries] = load_method1(cfg);
m2Pool = load_method2(cfg);
m2Map = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(m2Pool)
    m2Map(m2_key(m2Pool(i).plane_id, m2Pool(i).roi_idx)) = i;
end
candKeys = {};
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(weak)
    k = m2_key(weak(i).m2_plane, weak(i).m2_roi);
    if ~isKey(seen, k)
        candKeys{end+1} = k; %#ok<AGROW>
        seen(k) = true;
    end
end
for i = 1:numel(um2Rows)
    k = m2_key(um2Rows(i).m2_plane, um2Rows(i).m2_roi);
    if ~isKey(seen, k)
        candKeys{end+1} = k; %#ok<AGROW>
        seen(k) = true;
    end
end
trustKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(trusted)
    trustKeys(m2_key(trusted(i).m2_plane, trusted(i).m2_roi)) = true;
end
for i = 1:numel(m2Pool)
    k = m2_key(m2Pool(i).plane_id, m2Pool(i).roi_idx);
    if ~isKey(trustKeys, k) && ~isKey(seen, k)
        candKeys{end+1} = k; %#ok<AGROW>
        seen(k) = true;
    end
end
fprintf('bin-recover candidates: %d\n', numel(candKeys));

tfm = load_xy_tfm(matchDir);
calib = load_psf_coef(cfg);
coefXz = calib.coef_psf_xz;
coefYz = calib.coef_psf_yz;
factor = calib.factor;
x0 = calib.x0; y0 = calib.y0;
moveI = round(compute_move(calib.centers_1based, cfg.moveDivisor));
fprintf('\n=== open data.bin memmaps ===\n');
[mems, ly, lx, nf] = open_view_memmaps(cfg);
fprintf('FOV %dx%d T=%d stride=%d\n', ly, lx, nf, cfg.tStrideRecover);

results = [];
for k = 1:numel(candKeys)
    key = candKeys{k};
    if ~isKey(m2Map, key)
        fprintf('  skip %s (not in pool)\n', key);
        continue
    end
    m2 = m2Pool(m2Map(key));
    fprintf('  [%d/%d] p%d/roi%d ...\n', k, numel(candKeys), m2.plane_id, m2.roi_idx);
    res = recover_one(m2, tfm, mems, ly, lx, nf, moveI, coefXz, coefYz, factor, x0, y0, cfg);
    fprintf('      best_r=%.3f z=%.1f col=%.1f row=%.1f primary=v%d n_pass=%d -> %s\n', ...
        res.best_r, res.best_z, res.col, res.row, res.primary_v0 + 1, res.n_pass, ...
        ternary(res.accepted, 'ACCEPT', 'reject'));
    if isempty(results)
        results = res;
    else
        results(end+1) = res; %#ok<AGROW>
    end
end
nAcc = sum([results.accepted]);
fprintf('\naccepted recoveries %d / %d\n', nAcc, numel(results));
write_recovery_report(results, recDir);

recoveredEntries = [];
for i = 1:numel(results)
    if results(i).accepted
        recoveredEntries = [recoveredEntries; build_synthetic_entry(results(i), coefXz, coefYz, factor, x0, y0)]; %#ok<AGROW>
    end
end
if ~isempty(recoveredEntries)
    save(fullfile(recDir, 'recovered_neurons.mat'), 'recoveredEntries');
end
fprintf('\n=== merge m2 3D catalog ===\n');
merge_catalog(m2Pool, trusted, results, entriesM1, c1Orig, primaries, recoveredEntries, covDir, matchDir, cfg);
fprintf('\nDone. Originals untouched.\n');
end

function recover_trusted_only(cfg, matchDir, covDir)
matches = load_matches(fullfile(matchDir, 'matches.csv'));
[trusted, ~] = classify_matches(matches, cfg);
[entriesM1, c1Orig, primaries] = load_method1(cfg);
m2Pool = load_method2(cfg);
results = struct('m2', {}, 'accepted', {});
merge_catalog(m2Pool, trusted, results, entriesM1, c1Orig, primaries, [], covDir, matchDir, cfg);
end

function k = m2_key(plane, roi)
k = sprintf('%d_%d', plane, roi);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

function write_recovery_report(results, recDir)
rows = struct('m2_plane', {}, 'm2_roi', {}, 'col', {}, 'row', {}, 'z2', {}, 'best_z', {}, ...
    'best_r', {}, 'r_v1', {}, 'r_v2', {}, 'r_v3', {}, 'r_v4', {}, 'n_pass', {}, ...
    'accepted', {}, 'primary_view', {}, 'used_warped_fp', {});
for i = 1:numel(results)
    r = results(i);
    rows(i).m2_plane = r.m2.plane_id;
    rows(i).m2_roi = r.m2.roi_idx;
    rows(i).col = r.col;
    rows(i).row = r.row;
    rows(i).z2 = r.z2;
    rows(i).best_z = r.best_z;
    rows(i).best_r = r.best_r;
    rs = r.rs;
    rows(i).r_v1 = rs(1); rows(i).r_v2 = rs(2); rows(i).r_v3 = rs(3); rows(i).r_v4 = rs(4);
    rows(i).n_pass = r.n_pass;
    rows(i).accepted = double(r.accepted);
    rows(i).primary_view = r.primary_v0 + 1;
    rows(i).used_warped_fp = double(r.used_warped_fp);
end
write_csv(fullfile(recDir, 'recovery_report.csv'), rows, ...
    {'m2_plane', 'm2_roi', 'col', 'row', 'z2', 'best_z', 'best_r', 'r_v1', 'r_v2', 'r_v3', 'r_v4', ...
    'n_pass', 'accepted', 'primary_view', 'used_warped_fp'});
end

function merge_catalog(m2Pool, trusted, recoverResults, entriesM1, c1Orig, primaries, recoveredEntries, outDir, matchDir, cfg)
ensure_dir(outDir);
trustMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(trusted)
    trustMap(m2_key(trusted(i).m2_plane, trusted(i).m2_roi)) = i;
end
recMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(recoverResults)
    recMap(m2_key(recoverResults(i).m2.plane_id, recoverResults(i).m2.roi_idx)) = i;
end
recEntryMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(recoveredEntries)
    recEntryMap(m2_key(recoveredEntries(i).m2_plane, recoveredEntries(i).m2_roi)) = i;
end
rows = struct('m2_plane', {}, 'm2_roi', {}, 'source', {}, 'col', {}, 'row', {}, 'z_um', {}, ...
    'corr_or_best_r', {}, 'primary_view', {}, 'm1_idx', {}, 'd_xy_px', {}, 'd_z_um', {}, ...
    'n_pass', {}, 'accepted', {});
dictOut = {};
bySrc = struct();
for i = 1:numel(m2Pool)
    m2 = m2Pool(i);
    key = m2_key(m2.plane_id, m2.roi_idx);
    row = empty_cat_row(m2);
    if isKey(trustMap, key)
        m = trusted(trustMap(key));
        i1 = m.m1_idx + 1;
        row.source = 'matched_trusted';
        row.col = c1Orig(i1, 1);
        row.row = c1Orig(i1, 2);
        row.z_um = c1Orig(i1, 3);
        row.corr_or_best_r = m.corr;
        row.primary_view = primaries(i1);
        row.m1_idx = m.m1_idx;
        row.d_xy_px = m.d_xy_px;
        row.d_z_um = m.d_z_um;
        row.accepted = 1;
        e = entriesM1(i1);
        e.center = [row.col, row.row, row.z_um];
        e.source = 'matched_trusted';
        e.m2_plane = m2.plane_id;
        e.m2_roi = m2.roi_idx;
        e.m1_idx = m.m1_idx;
        dictOut{end+1} = e; %#ok<AGROW>
        bySrc = bump(bySrc, 'matched_trusted');
    elseif isKey(recMap, key) && recoverResults(recMap(key)).accepted && isKey(recEntryMap, key)
        r = recoverResults(recMap(key));
        row.source = 'recovered_bin';
        row.col = r.col;
        row.row = r.row;
        row.z_um = r.best_z;
        row.corr_or_best_r = r.best_r;
        row.primary_view = r.primary_v0 + 1;
        row.m1_idx = -1;
        row.d_z_um = abs(r.best_z - r.z2);
        row.n_pass = r.n_pass;
        row.accepted = 1;
        dictOut{end+1} = recoveredEntries(recEntryMap(key)); %#ok<AGROW>
        bySrc = bump(bySrc, 'recovered_bin');
    else
        row.source = 'failed';
        row.m1_idx = -1;
        row.accepted = 0;
        if isKey(recMap, key)
            r = recoverResults(recMap(key));
            row.corr_or_best_r = r.best_r;
            row.n_pass = r.n_pass;
        end
        bySrc = bump(bySrc, 'failed');
    end
    rows(i) = row;
end
fields = {'m2_plane', 'm2_roi', 'source', 'col', 'row', 'z_um', 'corr_or_best_r', ...
    'primary_view', 'm1_idx', 'd_xy_px', 'd_z_um', 'n_pass', 'accepted'};
write_csv(fullfile(outDir, 'm2_3d_catalog.csv'), rows, fields);
if ~isempty(dictOut)
    save(fullfile(outDir, 'm2_3d_dictionary.mat'), 'dictOut');
end
nOk = sum(~strcmp({rows.source}, 'failed'));
summary = struct('n_m2', numel(m2Pool), 'n_with_3d', nOk, ...
    'coverage', nOk / max(1, numel(m2Pool)), 'by_source', bySrc, ...
    'match_dir', matchDir, 'output_dir', outDir);
write_json(fullfile(outDir, 'coverage_summary.json'), summary);
fprintf('coverage: %d/%d (%.1f%%)\n', nOk, numel(m2Pool), 100 * nOk / max(1, numel(m2Pool)));
end

function row = empty_cat_row(m2)
row.m2_plane = m2.plane_id;
row.m2_roi = m2.roi_idx;
row.source = '';
row.col = '';
row.row = '';
row.z_um = '';
row.corr_or_best_r = '';
row.primary_view = '';
row.m1_idx = -1;
row.d_xy_px = '';
row.d_z_um = '';
row.n_pass = '';
row.accepted = 0;
end

function s = bump(s, name)
if ~isfield(s, name)
    s.(name) = 1;
else
    s.(name) = s.(name) + 1;
end
end
