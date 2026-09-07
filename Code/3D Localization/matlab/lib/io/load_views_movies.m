function views = load_views_movies(cfg)
%LOAD_VIEWS_MOVIES Pack (Y, X, T, 4) single from data.bin or TIFF fallback.
if nargin < 1, cfg = dataset_config(); end
tStride = max(cfg.tStride, 1);
opsList = cell(4, 1);
ly = [];
lx = [];
nAvg = inf;
for v = 1:4
    s2p = load_suite2p(view_plane_dir(v, cfg));
    lyV = s2p.ops.Ly;
    lxV = s2p.ops.Lx;
    nfV = s2p.ops.nframes;
    if isempty(ly)
        ly = lyV;
        lx = lxV;
    elseif lyV ~= ly || lxV ~= lx
        error('view %d size mismatch: %dx%d vs %dx%d', v, lyV, lxV, ly, lx);
    end
    nAvg = min(nAvg, floor(nfV / tStride));
    opsList{v} = s2p;
end
views = zeros(ly, lx, nAvg, 4, 'single');
for v = 1:4
    s2p = opsList{v};
    nfV = s2p.ops.nframes;
    binPath = view_bin_path(v, cfg);
    useTiff = ~bin_is_valid(binPath, nfV, ly, lx);
    if useTiff
        tiffPath = view_tiff_path(v, cfg);
        fprintf('  view%d: data.bin missing/blank - falling back to %s\n', v, tiffPath);
        movie = load_movie_from_tiff(tiffPath, ly, lx, tStride, s2p.ops.yoff, s2p.ops.xoff);
    else
        info = dir(binPath);
        nPix = ly * lx;
        if info.bytes == nfV * nPix * 2
            dtype = 'int16';
        elseif info.bytes == nfV * nPix * 4
            dtype = 'single';
        else
            error('unexpected data.bin size for view %d', v);
        end
        mm = open_bin_memmap(binPath, nfV, ly, lx, dtype);
        movie = average_bin_blocks(mm, tStride);
        fprintf('  view%d: averaged %d/%d frames from data.bin -> T=%d\n', ...
            v, nAvg * tStride, nfV, size(movie, 3));
    end
    if size(movie, 3) < nAvg
        error('view %d averaged T=%d < %d', v, size(movie, 3), nAvg);
    end
    views(:, :, :, v) = movie(:, :, 1:nAvg);
end
fprintf('  packed views %s\n', mat2str(size(views)));
end
