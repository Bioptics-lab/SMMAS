function run_full_pipeline(cfg)
%RUN_FULL_PIPELINE MATLAB TPLFM 4-view localization pipeline.
if nargin < 1
    cfg = dataset_config();
else
    addpath(fullfile(cfg.matlabDir, 'lib', 'io'));
    addpath(fullfile(cfg.matlabDir, 'lib', 'psf'));
    addpath(fullfile(cfg.matlabDir, 'lib', 'localize'));
    addpath(fullfile(cfg.matlabDir, 'lib', 'match'));
    addpath(fullfile(cfg.matlabDir, 'lib', 'recover'));
end
fprintf('PROJECT = %s\n', cfg.project);
fprintf('OUT_DIR = %s\n', cfg.outDir);
ensure_dir(cfg.outDir);
set(0, 'DefaultFigureVisible', 'off');

steps = {
    'prepare_view_bins_from_tiff'
    'run_localize'
    'match_to_recon_planes'
    'promote_unmatched'
    'recover_m2_from_bin'
    'refine_z_from_footprints'
    'dedupe_m2'
    'export_deduped_suite2p'
    'plot_m2_3d_distribution'
    };
for i = 1:numel(steps)
    name = steps{i};
    fprintf('\n======== %s ========\n', name);
    feval(name, cfg);
end
fprintf('\nAll pipeline steps finished.\n');
fprintf('MATLAB outputs: %s\n', cfg.outDir);
end
