function plot_m2_3d_distribution(cfg)
%PLOT_M2_3D_DISTRIBUTION 3D scatter + z histogram from m2_3d_catalog.csv.
if nargin < 1, cfg = dataset_config(); end
catalog = fullfile(cfg.outDir, 'match', 'm2_3d_coverage', 'm2_3d_catalog.csv');
if ~exist(catalog, 'file')
    error('missing %s', catalog);
end
outDir = fileparts(catalog);
rows = load_catalog_rows(catalog);
if isempty(rows)
    error('no neurons with 3D coords in %s', catalog);
end
x = [rows.col] * cfg.pixelUm;
y = [rows.row] * cfg.pixelUm;
z = [rows.z_um];
src = {rows.source};
zPlotMax = 30;
inZ = abs(z) <= zPlotMax;
nOmit = sum(~inZ);
xS = x(inZ); yS = y(inZ); zS = z(inZ);
fprintf('plotted %d / %d (omitted |z|>%.0f: %d)\n', sum(inZ), numel(rows), zPlotMax, nOmit);

fov = cfg.fovUm;
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 860 520]);
ax = axes(fig);
scatter3(ax, xS, yS, -zS, 60, [0.541 0.643 0.902], 'filled', ...
    'MarkerEdgeColor', [0.149 0.149 0.149], 'LineWidth', 0.5);
hold(ax, 'on');
draw_plane_frames(ax, cfg);
view(ax, 327.62631579 - 360 - 90, 19);
xlim(ax, [0 fov]); ylim(ax, [0 fov]); zlim(ax, [-25 25]);
xlabel(ax, 'Lateral position (\mum)');
ylabel(ax, 'Lateral position (\mum)');
zlabel(ax, 'Axial position (\mum)');
grid(ax, 'on');
p3d = fullfile(outDir, 'neurons_3d_scatter.png');
print(fig, p3d, '-dpng', '-r160');
close(fig);
fprintf('wrote %s n=%d (omitted |z|>%.0f: %d)\n', p3d, sum(inZ), zPlotMax, nOmit);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 420]);
zMin = floor(min(z) / 2) * 2;
zMax = ceil(max(z) / 2) * 2;
bins = zMin:2:zMax;
sources = {};
data = {};
colors = {};
for name = {'matched_trusted', 'recovered_bin_refined', 'recovered_bin'}
    mask = strcmp(src, name{1});
    if any(mask)
        sources{end+1} = sprintf('%s (n=%d)', name{1}, sum(mask)); %#ok<AGROW>
        data{end+1} = z(mask); %#ok<AGROW>
        if strcmp(name{1}, 'matched_trusted')
            colors{end+1} = [1 0.2 0.2]; %#ok<AGROW>
        else
            colors{end+1} = [0.541 0.643 0.902]; %#ok<AGROW>
        end
    end
end
if ~isempty(data)
    histogram(data{1}, bins, 'FaceColor', colors{1}, 'EdgeColor', 'k');
    hold on;
    for i = 2:numel(data)
        histogram(data{i}, bins, 'FaceColor', colors{i}, 'EdgeColor', 'k');
    end
    legend(sources, 'Location', 'northeast');
end
xlabel('Axial position (\mum)');
ylabel('neuron count');
title(sprintf('Neuron count along z (bin=2 \\mum, n=%d)', numel(rows)));
grid on;
ph = fullfile(outDir, 'neurons_z_hist.png');
print(fig, ph, '-dpng', '-r140');
close(fig);
fprintf('wrote %s\n', ph);
fprintf('z range [%.1f, %.1f] um  median=%.1f\n', min(z), max(z), median(z));
end

function rows = load_catalog_rows(path)
raw = read_csv(path);
rows = struct('col', {}, 'row', {}, 'z_um', {}, 'source', {}, 'plane', {}, 'roi', {});
k = 0;
for i = 1:numel(raw)
    src = csv_field(raw(i), 'source', '');
    if strcmp(src, 'failed'), continue; end
    col = csv_field(raw(i), 'col', []);
    zu = csv_field(raw(i), 'z_um', []);
    if isempty(col) || ~isnumeric(col) || isempty(zu) || ~isnumeric(zu)
        continue
    end
    k = k + 1;
    rows(k).col = col;
    rows(k).row = csv_field(raw(i), 'row', 0);
    rows(k).z_um = zu;
    rows(k).source = src;
    rows(k).plane = csv_field(raw(i), 'm2_plane', 0);
    rows(k).roi = csv_field(raw(i), 'm2_roi', 0);
end
end

function draw_plane_frames(ax, cfg)
%DRAW_PLANE_FRAMES Frames at -PLANE_Z_BASE (scatter uses -z).
fov = cfg.fovUm;
cols = [1 0.41 0.71; 0 0.55 0.55; 0.53 0.81 0.92; 0.56 0.74 0.56];
x0 = 0; x1 = fov; y0 = 0; y1 = fov;
for i = 1:numel(cfg.planeIds)
    z0 = -cfg.planeZBase(cfg.planeIds(i));
    c = cols(mod(i - 1, size(cols, 1)) + 1, :);
    plot3(ax, [x0 x1 x1 x0 x0], [y0 y0 y1 y1 y0], z0 * ones(1, 5), ...
        'Color', c, 'LineWidth', 1.6);
end
end
