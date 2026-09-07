function p = view_tiff_path(v, cfg)
if nargin < 2, cfg = dataset_config(); end
d = fileparts(fileparts(view_plane_dir(v, cfg))); % Views/{v}
p = fullfile(d, sprintf('file_%d.tif', v));
if ~exist(p, 'file')
    error('missing TIFF %s', p);
end
end
