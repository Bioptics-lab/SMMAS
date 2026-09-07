function matchDir = resolve_match_dir(cfg)
if nargin < 1, cfg = dataset_config(); end
root = fullfile(cfg.outDir, 'match');
loose = fullfile(root, 'loose');
base = fullfile(root, 'baseline');
if exist(fullfile(loose, 'matches.csv'), 'file')
    matchDir = loose;
elseif exist(fullfile(base, 'matches.csv'), 'file')
    matchDir = base;
else
    matchDir = fullfile(root, cfg.matchMode);
end
end
