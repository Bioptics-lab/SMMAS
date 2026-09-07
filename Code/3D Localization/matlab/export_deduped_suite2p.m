function export_deduped_suite2p(cfg)
%EXPORT_DEDUPED_SUITE2P Write kept plane traces/stat sidecars under MATLAB output.
if nargin < 1, cfg = dataset_config(); end
keptCsv = fullfile(cfg.outDir, 'match', 'm2_3d_coverage', 'dedupe', 'kept_rois.csv');
outRoot = fullfile(cfg.outDir, 'm2_suite2p_after_dedupe');
if ~exist(keptCsv, 'file')
    fprintf('missing %s; skip export\n', keptCsv);
    return
end
keptBy = load_kept(keptCsv, cfg);
ensure_dir(outRoot);
mapRows = struct('plane_id', {}, 'new_roi_idx', {}, 'old_roi_idx', {}, 'snr', {}, 'out_dir', {});
k = 0;
for p = 1:numel(cfg.planeIds)
    pid = cfg.planeIds(p);
    planeDir = plane_suite2p_dir(pid, cfg);
    planeName = sprintf('Plane%d', pid);
    outPlane0 = fullfile(outRoot, planeName, 'suite2p', 'plane0');
    kept = [];
    if isKey(keptBy, pid)
        kept = keptBy(pid);
    end
    rows = export_plane(planeDir, pid, kept, outPlane0, cfg);
    for i = 1:numel(rows)
        k = k + 1;
        mapRows(k) = rows(i);
    end
end
write_csv(fullfile(outRoot, 'roi_index_map.csv'), mapRows, ...
    {'plane_id', 'new_roi_idx', 'old_roi_idx', 'snr', 'out_dir'});
readme = {
    'Deduped method2 suite2p exports (cross-plane duplicates removed; higher SNR kept).'
    'Original suite2p folders were NOT modified.'
    ''
    'Layout:'
    };
for p = 1:numel(cfg.planeIds)
    readme{end+1} = sprintf('  Plane%d/suite2p/plane0/', cfg.planeIds(p)); %#ok<AGROW>
end
readme{end+1} = ''; %#ok<AGROW>
readme{end+1} = 'Each plane0 contains numeric F.npy, Fneu.npy, iscell.npy and s2p_matlab/ sidecars.'; %#ok<AGROW>
readme{end+1} = 'ROI indices are reindexed 0..N-1 per plane; see roi_index_map.csv for old_roi_idx.'; %#ok<AGROW>
readme{end+1} = sprintf('Total kept ROIs: %d', numel(mapRows)); %#ok<AGROW>
fid = fopen(fullfile(outRoot, 'README.txt'), 'w');
fprintf(fid, '%s\n', readme{:});
fclose(fid);
fprintf('wrote %s\n', fullfile(outRoot, 'roi_index_map.csv'));
fprintf('OUT_ROOT = %s  total kept %d\n', outRoot, numel(mapRows));
end

function keptBy = load_kept(keptCsv, cfg)
keptBy = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:numel(cfg.planeIds)
    keptBy(cfg.planeIds(i)) = zeros(0, 2);
end
rows = read_csv(keptCsv);
for i = 1:numel(rows)
    pid = rows(i).plane_id;
    pair = [rows(i).roi_idx, rows(i).snr];
    if isKey(keptBy, pid)
        keptBy(pid) = [keptBy(pid); pair];
    else
        keptBy(pid) = pair;
    end
end
ks = keptBy.keys;
for i = 1:numel(ks)
    arr = keptBy(ks{i});
    if ~isempty(arr)
        [~, ord] = sort(arr(:, 1));
        keptBy(ks{i}) = arr(ord, :);
    end
end
end

function rows = export_plane(planeDir, planeId, kept, outPlane0, cfg)
rows = struct('plane_id', {}, 'new_roi_idx', {}, 'old_roi_idx', {}, 'snr', {}, 'out_dir', {});
if isempty(kept)
    fprintf('plane %d: no kept ROIs, skip\n', planeId);
    return
end
s2p = load_suite2p(planeDir);
oldIdx0 = kept(:, 1); % 0-based
oldIdx = oldIdx0 + 1;
F2 = s2p.F(oldIdx, :);
Fneu2 = s2p.Fneu(oldIdx, :);
iscell2 = ones(numel(oldIdx), 2);
ensure_dir(outPlane0);
writeNPY(single(F2), fullfile(outPlane0, 'F.npy'));
writeNPY(single(Fneu2), fullfile(outPlane0, 'Fneu.npy'));
writeNPY(iscell2, fullfile(outPlane0, 'iscell.npy'));
metaSrc = fullfile(planeDir, 's2p_matlab');
metaDst = fullfile(outPlane0, 's2p_matlab');
ensure_dir(metaDst);
xpix = []; ypix = []; lam = [];
off = []; cnt = [];
med = zeros(numel(oldIdx), 2);
srcOff = readNPY(fullfile(metaSrc, 'pix_offset.npy'));
srcCnt = readNPY(fullfile(metaSrc, 'pix_count.npy'));
srcX = readNPY(fullfile(metaSrc, 'xpix.npy'));
srcY = readNPY(fullfile(metaSrc, 'ypix.npy'));
srcL = readNPY(fullfile(metaSrc, 'lam.npy'));
srcMed = readNPY(fullfile(metaSrc, 'med.npy'));
for i = 1:numel(oldIdx)
    oi = oldIdx(i);
    a = srcOff(oi) + 1;
    b = srcOff(oi) + srcCnt(oi);
    off(i, 1) = numel(xpix); %#ok<AGROW>
    cnt(i, 1) = srcCnt(oi); %#ok<AGROW>
    if srcCnt(oi) >= 1
        xpix = [xpix; srcX(a:b)]; %#ok<AGROW>
        ypix = [ypix; srcY(a:b)]; %#ok<AGROW>
        lam = [lam; srcL(a:b)]; %#ok<AGROW>
    end
    if size(srcMed, 1) >= oi
        med(i, :) = srcMed(oi, :);
    end
end
writeNPY(xpix, fullfile(metaDst, 'xpix.npy'));
writeNPY(ypix, fullfile(metaDst, 'ypix.npy'));
writeNPY(lam, fullfile(metaDst, 'lam.npy'));
writeNPY(int32(off), fullfile(metaDst, 'pix_offset.npy'));
writeNPY(int32(cnt), fullfile(metaDst, 'pix_count.npy'));
writeNPY(med, fullfile(metaDst, 'med.npy'));
copyfile(fullfile(metaSrc, 'meanImg.npy'), fullfile(metaDst, 'meanImg.npy'));
opsJ = jsondecode(fileread(fullfile(metaSrc, 'ops.json')));
opsJ.nroi = numel(oldIdx);
write_json(fullfile(metaDst, 'ops.json'), opsJ);
rel = ['m2_suite2p_after_dedupe/Plane_' num2str(planeId) '_motionfree/suite2p/plane0'];
for i = 1:numel(oldIdx)
    rows(i).plane_id = planeId;
    rows(i).new_roi_idx = i - 1;
    rows(i).old_roi_idx = oldIdx0(i);
    rows(i).snr = kept(i, 2);
    rows(i).out_dir = rel;
end
fprintf('plane %d: kept %d / orig %d -> %s\n', planeId, numel(oldIdx), numel(s2p.stat), outPlane0);
end
