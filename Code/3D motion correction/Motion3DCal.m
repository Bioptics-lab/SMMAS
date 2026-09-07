%% 3D motion reconstruction for SMMAS data

%% PSF processing
% Parameters
XYcenter = [143, 114];
zcenter = 30;
Reconstruct_range = 40;
zrange = (zcenter - Reconstruct_range / 2):1:(zcenter + Reconstruct_range / 2);

% Load the four-view PSF stacks, segment, and fit shear coefficients
path = 'PSF_stack';
filename = [path '/file_1.tif';
            path '/file_2.tif';
            path '/file_3.tif';
            path '/file_4.tif'];
info_PSF = imfinfo(filename(1, :));
num_images = numel(info_PSF);
psf_lens = [];
k_xz = zeros(4, 2);
k_yz = zeros(4, 2);
k_zz = ones(4, 2);
for views = 1:4
    psf_view = loadtiff(filename(views, :));
    [psf_view_current, k_xz(views, :), k_yz(views, :)] = psf_seg_TPLFM_fit(psf_view, XYcenter, zrange);
    psf_lens = cat(4, psf_lens, psf_view_current);
end

% Save the segmented PSF
psfname = [path '/PSF_1230_move.mat'];
save(psfname, 'psf_lens');

%% Data and PSF pixel size
PSF_zoom = 15;
Image_zoom = 5;
PSF_pixel = 256;
Image_pixel = 256;
resizefactor = (PSF_zoom / Image_zoom) * (PSF_pixel / Image_pixel);

%% Data preprocessing
% Convert PSF shear coefficients to the imaging pixel scale
k_xz = k_xz / resizefactor;
k_yz = k_yz / resizefactor;
x_motion = [];
y_motion = [];
load('NoRMcorreOutput/View1Motion.mat');
x_motion(:, 1) = mean(shifts_x, 2);
y_motion(:, 1) = mean(shifts_y, 2);
load('NoRMcorreOutput/View2Motion.mat');
x_motion(:, 2) = mean(shifts_x, 2);
y_motion(:, 2) = mean(shifts_y, 2);
load('NoRMcorreOutput/View3Motion.mat');
x_motion(:, 3) = mean(shifts_x, 2);
y_motion(:, 3) = mean(shifts_y, 2);
load('NoRMcorreOutput/View4Motion.mat');
x_motion(:, 4) = mean(shifts_x, 2);
y_motion(:, 4) = mean(shifts_y, 2);

%% 3D motion reconstruction
Delta = [];
Res = [];
for i = 1:6000
    [Delta(:, i), Res(:, i)] = Motion3Dsolve(x_motion(i, :), y_motion(i, :), k_xz(:, 1), k_yz(:, 1));
end

%% Plot results
% Apparent x motion in each view
figure;
plot(x_motion(:, 1), 'LineWidth', 1.5, 'color', [1 0.5 0.5]); hold on;   % red
plot(x_motion(:, 2), 'LineWidth', 1.5, 'color', [0.5 1 0.5]); hold on;   % green
plot(x_motion(:, 4), 'LineWidth', 1.5, 'color', [0.5 0.5 1]);            % blue
plot(x_motion(:, 3), 'LineWidth', 1.5, 'color', [0.8 0.8 0.1]); hold on; % yellow

hLegend = legend('View 1 (+0 ns)', 'View 2 (+3.1 ns)', 'View 4 (+6.2 ns)', 'View 3 (+9.3 ns)');
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(hLegend, 'FontSize', get(hLegend, 'FontSize') * 2); % double legend font size
set(gca, 'LineWidth', get(gca, 'LineWidth') * 2);       % double axes line width

% Apparent y motion in each view
figure;
plot(y_motion(:, 1), 'LineWidth', 1.5, 'color', [1 0.5 0.5]); hold on;   % red
plot(y_motion(:, 2), 'LineWidth', 1.5, 'color', [0.5 1 0.5]); hold on;   % green
plot(y_motion(:, 4), 'LineWidth', 1.5, 'color', [0.5 0.5 1]);            % blue
plot(y_motion(:, 3), 'LineWidth', 1.5, 'color', [0.8 0.8 0.1]); hold on; % yellow

hLegend = legend('View 1 (+0 ns)', 'View 2 (+3.1 ns)', 'View 4 (+6.2 ns)', 'View 3 (+9.3 ns)');
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(hLegend, 'FontSize', get(hLegend, 'FontSize') * 2); % double legend font size
set(gca, 'LineWidth', get(gca, 'LineWidth') * 2);       % double axes line width

% First residual component
figure;
plot(Res(1, :));

% Overlay of y motion from all four views
figure;
plot(y_motion(:, 1)); hold on;
plot(y_motion(:, 2)); hold on;
plot(y_motion(:, 3)); hold on;
plot(y_motion(:, 4));

% Scale and flip the reconstructed motion, then plot
Delta(1, :) = Delta(1, :) * 255 / 256 * 1.4;
Delta(2, :) = Delta(2, :) * 255 / 256 * 1.4;
Delta(3, :) = -Delta(3, :) * 2;
figure;
plot(Delta(1, :), 'LineWidth', 1.5); hold on;
plot(Delta(2, :), 'LineWidth', 1.5); hold on;
plot(Delta(3, :), 'LineWidth', 1.5);

hLegend = legend('Reconstructed x motion', 'Reconstructed y motion', 'Reconstructed z motion');
set(hLegend, 'FontSize', get(hLegend, 'FontSize') * 2); % double legend font size
set(gca, 'LineWidth', get(gca, 'LineWidth') * 2);       % double axes line width
set(hLegend, 'Location', 'northeast');                  % place legend at the upper right
ylim([min(Delta(:)) - 1, max(Delta(:)) + 3]);          % set y-axis limits manually
