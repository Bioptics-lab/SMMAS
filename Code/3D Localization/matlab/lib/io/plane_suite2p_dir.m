function p = plane_suite2p_dir(planeId, cfg)
if nargin < 2, cfg = dataset_config(); end
p = fullfile(cfg.planesDir, sprintf('Plane%d', planeId), 'suite2p', 'plane0');
end
