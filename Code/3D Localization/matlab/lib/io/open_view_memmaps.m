function [mems, ly, lx, nf] = open_view_memmaps(cfg)
%OPEN_VIEW_MEMMAPS Open four view data.bin memmaps. Errors if a bin is blank.
if nargin < 1, cfg = dataset_config(); end
mems = cell(4, 1);
ly = [];
lx = [];
nf = [];
for v = 1:4
    s2p = load_suite2p(view_plane_dir(v, cfg));
    lyV = s2p.ops.Ly;
    lxV = s2p.ops.Lx;
    nfV = s2p.ops.nframes;
    if isempty(ly)
        ly = lyV; lx = lxV; nf = nfV;
    elseif lyV ~= ly || lxV ~= lx || nfV ~= nf
        error('view %d shape/frames mismatch', v);
    end
    binPath = view_bin_path(v, cfg);
    if ~bin_is_valid(binPath, nfV, lyV, lxV)
        error('view %d data.bin is missing or blank', v);
    end
    info = dir(binPath);
    nPix = ly * lx;
    if info.bytes == nfV * nPix * 2
        dtype = 'int16';
    else
        dtype = 'single';
    end
    mems{v} = open_bin_memmap(binPath, nfV, lyV, lxV, dtype);
    fprintf('  view%d memmap %s frames=%d\n', v, binPath, nfV);
end
end
