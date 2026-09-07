function calib = load_psf_coef(cfg)
%LOAD_PSF_COEF Load psf_fit_coef.mat written by run_localize.
if nargin < 1, cfg = dataset_config(); end
p = fullfile(cfg.outDir, 'psf_fit_coef.mat');
if ~exist(p, 'file')
    error('missing %s', p);
end
calib = load(p);
end
