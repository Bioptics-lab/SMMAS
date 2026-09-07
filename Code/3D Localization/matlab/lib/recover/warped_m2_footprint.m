function fp = warped_m2_footprint(m2, tfm, h, w)
%WARPED_M2_FOOTPRINT Inverse-warp plane ROI pixels into method1 FOV (0-based).
xpix = double(m2.xpix(:));
ypix = double(m2.ypix(:));
if numel(xpix) < 3
    fp = [];
    return
end
[col, row] = inverse_xy(xpix, ypix, tfm);
xs = round(col);
ys = round(row);
ok = xs >= 0 & xs < w & ys >= 0 & ys < h;
xs = xs(ok);
ys = ys(ok);
if numel(xs) < 3
    fp = [];
    return
end
uv = unique([xs, ys], 'rows');
fp.xs = int32(uv(:, 1));
fp.ys = int32(uv(:, 2));
end
