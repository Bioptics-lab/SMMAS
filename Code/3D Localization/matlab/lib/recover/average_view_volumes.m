function vols = average_view_volumes(mems, avgBin)
%AVERAGE_VIEW_VOLUMES Block-mean each view to (Y, X, nAvg) single.
nf = mems{1}.nf;
ly = mems{1}.ly;
lx = mems{1}.lx;
nAvg = floor(nf / avgBin);
vols.vols = cell(1, 4);
vols.n_avg = nAvg;
vols.ly = ly;
vols.lx = lx;
vols.avg_bin = avgBin;
for v = 1:4
    fprintf('  average view%d: T=%d -> %d (bin=%d) ...\n', v, nf, nAvg, avgBin);
    out = zeros(ly, lx, nAvg, 'single');
    for i = 1:nAvg
        acc = zeros(ly, lx);
        s = (i - 1) * avgBin;
        for j = 1:avgBin
            acc = acc + read_bin_frame(mems{v}, s + j);
        end
        out(:, :, i) = single(acc / avgBin);
    end
    vols.vols{v} = out;
    fprintf('    done view%d\n', v);
end
end
