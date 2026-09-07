function p = view_working_plane_dir(v, cfg)
%VIEW_WORKING_PLANE_DIR MATLAB-output copy of view suite2p (does not rewrite Views/).
if nargin < 2, cfg = dataset_config(); end
p = fullfile(cfg.outDir, 'views_suite2p_working', sprintf('view_%d', v), 'suite2p', 'plane0');
end

