function calib = calibrate_psf(cfg)
%CALIBRATE_PSF Segment PSF volumes, energy-balance, fit independent x(z)/y(z).
if nargin < 1, cfg = dataset_config(); end
ensure_dir(cfg.outDir);
segList = cell(1, 4);
centers1 = cfg.viewCenters;
volumes = cell(1, 4);
for v = 1:4
    path = fullfile(cfg.psfDir, sprintf('file_%d.tif', v));
    if ~exist(path, 'file')
        path = fullfile(cfg.project, sprintf('file_%d.tif', v));
    end
    fprintf('Loading PSF %s ...\n', path);
    volumes{v} = load_tiff_volume(path);
    fprintf('  shape (Y,X,Z)=%s\n', mat2str(size(volumes{v})));
end
zrange = cfg.psfZrange;
fprintf('PSF z (1-based) = %d:%d  (%d planes)  crop %dx%d\n', ...
    zrange(1), zrange(end), numel(zrange), 2 * cfg.cropHalfY + 1, 2 * cfg.cropHalfX + 1);
for v = 1:4
    center = centers1(v, :);
    fprintf('  view%d center(1-based row,col)=[%d %d] crop_half_y=%d crop_half_x=%d\n', ...
        v, center(1), center(2), cfg.cropHalfY, cfg.cropHalfX);
    seg = psf_seg_TPLFM(volumes{v}, center, zrange, cfg.cropHalfY, cfg.cropHalfX);
    segList{v} = seg;
    write_tiff_vol(fullfile(cfg.outDir, sprintf('psf%d_seg.tif', v)), seg);
end
[balanced, xpsf] = apply_energy_modification(segList);
for v = 1:4
    write_tiff_vol(fullfile(cfg.outDir, sprintf('psf%d_energy.tif', v)), balanced{v});
end
nZ = size(balanced{1}, 3);
zUm = ((0:nZ-1) - (nZ - 1) / 2) * cfg.psfDzUm;
psfParts = cat(4, balanced{:});
[fitLine, coefXz, coefYz, coefXy, x0, y0] = psf_fit(psfParts, zUm);
calib.fit_line = fitLine;
calib.coef_psf_xz = coefXz;
calib.coef_psf_yz = coefYz;
calib.coef_psf_xy = coefXy;
calib.x0 = x0;
calib.y0 = y0;
calib.xpsf = xpsf;
calib.z_um = zUm;
calib.centers_1based = centers1;
calib.crop_half_y = cfg.cropHalfY;
calib.crop_half_x = cfg.cropHalfX;
calib.dz_um = cfg.psfDzUm;
calib.factor = cfg.factor;
calib.move_divisor = cfg.moveDivisor;
save(fullfile(cfg.outDir, 'psf_fit_coef.mat'), '-struct', 'calib');
fprintf('coef_psf_xz slopes: %s\n', mat2str(coefXz(1, :), 4));
fprintf('coef_psf_yz slopes: %s\n', mat2str(coefYz(1, :), 4));
fprintf('x0=%.2f y0=%.2f  z_um=[%.1f, %.1f]\n', x0, y0, zUm(1), zUm(end));
plot_psf_trajectories(fitLine, zUm, coefXz, coefYz, cfg.outDir);
end

function plot_psf_trajectories(fitLine, zUm, coefXz, coefYz, outDir)
fig = figure('Visible', 'off');
for v = 1:4
    subplot(2, 2, v);
    plot(zUm, squeeze(fitLine(1, :, v)), 'o-', 'MarkerSize', 3); hold on;
    plot(zUm, squeeze(fitLine(2, :, v)), 's-', 'MarkerSize', 3);
    plot(zUm, polyval(coefXz(:, v), zUm), '--');
    plot(zUm, polyval(coefYz(:, v), zUm), '--');
    title(sprintf('view %d', v));
    xlabel('z (um)');
    ylabel('centroid (px, 1-based)');
    grid on;
    legend('x (col)', 'y (row)', 'Location', 'best');
end
sgtitle('PSF centroid trajectories (TPLFM LinearRunner)');
path = fullfile(outDir, 'psf_fit_trajectories.png');
print(fig, path, '-dpng', '-r120');
close(fig);
fprintf('wrote %s\n', path);
end
