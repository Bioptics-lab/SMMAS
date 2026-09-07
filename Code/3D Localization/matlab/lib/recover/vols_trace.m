function tr = vols_trace(vols, v, xs, ys)
n = vols.n_avg;
if isempty(xs)
    tr = zeros(1, n);
    return
end
xs = double(xs(:));
ys = double(ys(:));
ok = xs >= 0 & xs < vols.lx & ys >= 0 & ys < vols.ly;
xs = xs(ok); ys = ys(ok);
if numel(xs) < 3
    tr = zeros(1, n);
    return
end
if numel(xs) > 64
    sel = unique(round(linspace(1, numel(xs), 64)));
    xs = xs(sel); ys = ys(sel);
end
acc = zeros(1, n);
for p = 1:numel(xs)
    acc = acc + double(reshape(vols.vols{v}(ys(p) + 1, xs(p) + 1, :), 1, n));
end
tr = acc / numel(xs);
end
