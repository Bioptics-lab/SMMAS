function z = zscore_vec(tr)
tr = double(tr(:));
s = std(tr);
if s < 1e-12
    z = zeros(size(tr));
else
    z = (tr - mean(tr)) / s;
end
end
