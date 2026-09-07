function movie = average_bin_blocks(mm, tStride)
%AVERAGE_BIN_BLOCKS Block-average memmap frames to (Y, X, nAvg) single.
nf = mm.nf;
ly = mm.ly;
lx = mm.lx;
tStride = max(int32(tStride), 1);
nAvg = floor(nf / double(tStride));
movie = zeros(ly, lx, nAvg, 'single');
for i = 1:nAvg
    acc = zeros(ly, lx);
    s = (i - 1) * double(tStride);
    for j = 1:double(tStride)
        acc = acc + read_bin_frame(mm, s + j);
    end
    movie(:, :, i) = single(acc / double(tStride));
end
end
