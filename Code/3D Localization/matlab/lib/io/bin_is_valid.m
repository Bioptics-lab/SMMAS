function ok = bin_is_valid(binPath, nf, ly, lx)
%BIN_IS_VALID True if int16 data.bin has expected size and is not all zeros.
ok = false;
expect = nf * ly * lx * 2;
if ~exist(binPath, 'file')
    return
end
info = dir(binPath);
if info.bytes ~= expect
    return
end
mm = open_bin_memmap(binPath, nf, ly, lx, 'int16');
try
    probes = unique(max(1, min(nf, [1, floor(nf/4)+1, floor(nf/2)+1, floor(3*nf/4)+1, nf])));
    for i = 1:numel(probes)
        fr = read_bin_frame(mm, probes(i));
        if max(abs(fr(:))) ~= 0
            ok = true;
            return
        end
    end
catch
    ok = false;
end
end
