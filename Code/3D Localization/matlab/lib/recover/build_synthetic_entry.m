function e = build_synthetic_entry(res, coefXz, coefYz, factor, x0, y0)
p0 = res.primary_v0;
z = res.best_z;
col = res.col;
row = res.row;
[dxP, dyP] = parallax_dx_dy(z, p0, coefXz, coefYz, factor, x0, y0);
cav = zeros(2, 4);
cav(1, p0 + 1) = col;
cav(2, p0 + 1) = row;
for v0 = 0:3
    if v0 == p0, continue; end
    [dxV, dyV] = parallax_dx_dy(z, v0, coefXz, coefYz, factor, x0, y0);
    cav(1, v0 + 1) = col + (dxV - dxP);
    cav(2, v0 + 1) = row + (dyV - dyP);
end
[xs, ys] = disk_pixels(col, row, 5, 10000, 10000);
e.center = [col, row, z];
e.center_allview = cav;
e.pixels1 = double(xs) - col;
e.pixels2 = double(ys) - row;
e.primary_view = p0 + 1;
e.peak_corr = res.best_r;
e.source = 'recovered_from_bin';
e.m2_plane = res.m2.plane_id;
e.m2_roi = res.m2.roi_idx;
e.recovery_rs = res.rs(:)';
if isfield(res, 'traces')
    e.trace_primary = res.traces(p0 + 1, :);
else
    e.trace_primary = [];
end
end
