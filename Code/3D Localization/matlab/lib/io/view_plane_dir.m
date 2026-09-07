function p = view_plane_dir(v, cfg)
if nargin < 2, cfg = dataset_config(); end
cands = {fullfile(cfg.dataRoot, num2str(v)), fullfile(cfg.dataRoot, sprintf('file_%d', v))};
for i = 1:numel(cands)
    d = fullfile(cands{i}, 'suite2p', 'plane0');
    if exist(d, 'dir')
        p = d;
        return
    end
end
error('no suite2p plane0 for view %d under %s', v, cfg.dataRoot);
end
