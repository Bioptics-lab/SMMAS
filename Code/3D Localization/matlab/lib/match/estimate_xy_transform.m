function tfm = estimate_xy_transform(img1, img2)
%ESTIMATE_XY_TRANSFORM Uniform scale + translation mapping method1 -> method2.
h1 = size(img1, 1); w1 = size(img1, 2);
h2 = size(img2, 1); w2 = size(img2, 2);
ref = double(img2);
ref = (ref - mean(ref(:))) / (std(ref(:)) + 1e-12);
s0 = w2 / w1;
scales = unique(round([s0, linspace(s0 * 0.85, s0 * 1.15, 9), linspace(0.70, 0.95, 6)] * 1e4) / 1e4);
tfm = struct('s', s0, 'tx', (w2 - s0 * w1) / 2, 'ty', (h2 - s0 * h1) / 2, 'ncc', -2);
for si = 1:numel(scales)
    s = scales(si);
    scaled = imresize(double(img1), s, 'bilinear');
    hs = size(scaled, 1);
    ws = size(scaled, 2);
    if hs >= h2 && ws >= w2
        cy = floor((hs - h2) / 2);
        cx = floor((ws - w2) / 2);
        patch = scaled(cy + 1:cy + h2, cx + 1:cx + w2);
        mov = (patch - mean(patch(:))) / (std(patch(:)) + 1e-12);
        [dx, dy, ncc] = phase_shift(ref, mov);
        tx = -double(cx) + dx;
        ty = -double(cy) + dy;
    else
        canvas = zeros(h2, w2);
        y0 = max(0, floor((h2 - hs) / 2));
        x0 = max(0, floor((w2 - ws) / 2));
        y1 = min(h2, y0 + hs);
        x1 = min(w2, x0 + ws);
        sy0 = 0;
        sx0 = 0;
        if y0 < 0, sy0 = -y0; y0 = 0; end
        if x0 < 0, sx0 = -x0; x0 = 0; end
        canvas(y0 + 1:y1, x0 + 1:x1) = scaled(sy0 + 1:sy0 + (y1 - y0), sx0 + 1:sx0 + (x1 - x0));
        mov = (canvas - mean(canvas(:))) / (std(canvas(:)) + 1e-12);
        [dx, dy, ncc] = phase_shift(ref, mov);
        tx = double(x0 + dx);
        ty = double(y0 + dy);
    end
    if ncc > tfm.ncc
        tfm = struct('s', s, 'tx', tx, 'ty', ty, 'ncc', ncc);
    end
end
if tfm.ncc < 0.05
    s = s0;
    tfm.s = s;
    tfm.tx = (w2 - s * w1) / 2;
    tfm.ty = (h2 - s * h1) / 2;
    fprintf('XY reg: low NCC=%.3f — fallback s=%.4f center-align\n', tfm.ncc, s);
else
    fprintf('XY reg: s=%.4f  tx=%.2f  ty=%.2f  NCC=%.3f\n', tfm.s, tfm.tx, tfm.ty, tfm.ncc);
end
end

function [dx, dy, ncc] = phase_shift(ref, mov)
corr = conv2(ref, rot90(mov, 2), 'same');
[~, ind] = max(corr(:));
[py, px] = ind2sub(size(corr), ind);
h2 = size(ref, 1);
w2 = size(ref, 2);
dy = double(py - floor(h2 / 2) - 1);
dx = double(px - floor(w2 / 2) - 1);
shifted = circshift(mov, [dy, dx]);
ncc = ncc_images(ref, shifted);
end
