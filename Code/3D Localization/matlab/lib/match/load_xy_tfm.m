function tfm = load_xy_tfm(matchDir)
p = fullfile(matchDir, 'xy_transform.mat');
if ~exist(p, 'file')
    p = fullfile(fileparts(matchDir), 'baseline', 'xy_transform.mat');
end
if ~exist(p, 'file')
    error('missing xy_transform.mat under %s', matchDir);
end
tfm = load(p);
end
