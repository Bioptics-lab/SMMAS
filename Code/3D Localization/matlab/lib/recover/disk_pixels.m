function [xs, ys] = disk_pixels(cx, cy, radius, h, w)
r = max(1, round(radius));
rr2 = double(r * r);
xs = [];
ys = [];
for yi = floor(cy - r):ceil(cy + r)
    for xi = floor(cx - r):ceil(cx + r)
        if xi >= 0 && xi < w && yi >= 0 && yi < h && (xi - cx)^2 + (yi - cy)^2 <= rr2
            xs(end+1, 1) = xi; %#ok<AGROW>
            ys(end+1, 1) = yi; %#ok<AGROW>
        end
    end
end
if isempty(xs)
    xs = min(max(round(cx), 0), w - 1);
    ys = min(max(round(cy), 0), h - 1);
end
xs = int32(xs);
ys = int32(ys);
end
