function mm = open_bin_memmap(binPath, nf, ly, lx, dtype)
%OPEN_BIN_MEMMAP Map C-order (nframes, Ly, Lx) binary as MATLAB [Lx, Ly, Nf].
if nargin < 5, dtype = 'int16'; end
if strcmp(dtype, 'int16')
    fmt = 'int16';
elseif strcmp(dtype, 'single') || strcmp(dtype, 'float32')
    fmt = 'single';
else
    error('unsupported dtype %s', dtype);
end
mm.map = memmapfile(binPath, 'Format', {fmt, [lx, ly, nf], 'vol'}, 'Writable', false);
mm.ly = ly;
mm.lx = lx;
mm.nf = nf;
end
