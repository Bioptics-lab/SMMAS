function tr = block_average_trace(tr, avgBin, nAvg)
tr = double(tr(:))';
need = nAvg * avgBin;
if numel(tr) < need
    pad = zeros(1, need);
    pad(1:numel(tr)) = tr;
    tr = pad;
else
    tr = tr(1:need);
end
tr = mean(reshape(tr, avgBin, nAvg), 1);
end
