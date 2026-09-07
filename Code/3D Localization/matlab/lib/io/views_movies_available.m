function ok = views_movies_available(cfg)
%VIEWS_MOVIES_AVAILABLE True if every view has a valid data.bin or a TIFF.
if nargin < 1, cfg = dataset_config(); end
ok = true;
for v = 1:4
    s2p = load_suite2p(view_plane_dir(v, cfg));
    if bin_is_valid(view_bin_path(v, cfg), s2p.ops.nframes, s2p.ops.Ly, s2p.ops.Lx)
        continue
    end
    d = fileparts(fileparts(view_plane_dir(v, cfg)));
    tiffPath = fullfile(d, sprintf('file_%d.tif', v));
    if ~exist(tiffPath, 'file')
        ok = false;
        return
    end
end
end
