function p = view_bin_path(v, cfg)
if nargin < 2, cfg = dataset_config(); end
p = fullfile(view_plane_dir(v, cfg), 'data.bin');
end
