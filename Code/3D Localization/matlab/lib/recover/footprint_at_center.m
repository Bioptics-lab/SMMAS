function [xs, ys] = footprint_at_center(seedXs, seedYs, cx, cy, h, w)
if ~isempty(seedXs) && ~isempty(seedYs)
    scx = mean(double(seedXs));
    scy = mean(double(seedYs));
    xs = round(double(seedXs) - scx + cx);
    ys = round(double(seedYs) - scy + cy);
    ok = xs >= 0 & xs < w & ys >= 0 & ys < h;
    xs = xs(ok);
    ys = ys(ok);
    if numel(xs) >= 3
        xs = int32(xs);
        ys = int32(ys);
        return
    end
end
[xs, ys] = disk_pixels(cx, cy, 5, h, w);
end
