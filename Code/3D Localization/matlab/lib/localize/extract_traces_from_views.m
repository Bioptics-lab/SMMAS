function traces = extract_traces_from_views(merged, views)
%EXTRACT_TRACES_FROM_VIEWS (N, 4, T) mean traces at center_allview + footprint.
n = numel(merged);
[h, w, nT, nViews] = size(views);
traces = zeros(n, nViews, nT);
for i = 1:n
    traces(i, :, :) = extract_one(merged(i), views, h, w, nT, nViews);
end
end

function tr = extract_one(entry, views, h, w, nT, nViews)
cav = reshape(double(entry.center_allview), 2, 4);
d1 = double(entry.pixels1(:));
d2 = double(entry.pixels2(:));
tr = zeros(nViews, nT);
for v = 1:nViews
    cx = cav(1, v); cy = cav(2, v);
    if ~isfinite(cx) || ~isfinite(cy) || isempty(d1)
        continue
    end
    px = min(max(round(d1 + cx) + 1, 1), w);
    py = min(max(round(d2 + cy) + 1, 1), h);
    acc = zeros(1, nT);
    for p = 1:numel(px)
        acc = acc + double(reshape(views(py(p), px(p), :, v), 1, nT));
    end
    tr(v, :) = acc / numel(px);
end
end
