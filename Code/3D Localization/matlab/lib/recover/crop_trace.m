function tr = crop_trace(crops, v, xsAbs, ysAbs)
n = crops.nT;
y0 = crops.origins(v, 1);
x0 = crops.origins(v, 2);
bh = size(crops.vols{v}, 1);
bw = size(crops.vols{v}, 2);
xs = double(xsAbs(:)) - x0;
ys = double(ysAbs(:)) - y0;
ok = xs >= 0 & xs < bw & ys >= 0 & ys < bh;
if sum(ok) < 3
    tr = zeros(1, n);
    return
end
xs = xs(ok);
ys = ys(ok);
if numel(xs) > 80
    sel = unique(round(linspace(1, numel(xs), 80)));
    xs = xs(sel);
    ys = ys(sel);
end
xi = xs + 1;
yi = ys + 1;
acc = zeros(1, n);
for p = 1:numel(xi)
    acc = acc + double(reshape(crops.vols{v}(yi(p), xi(p), :), 1, n));
end
tr = acc / numel(xi);
end
