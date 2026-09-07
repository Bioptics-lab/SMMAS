function crops = make_crop_views(mems, boxes, frameIdx)
%MAKE_CROP_VIEWS Copy per-view search boxes at time indices frameIdx (1-based).
% boxes(v,:) = [y0 y1 x0 x1] 0-based half-open, matching Python search_boxes.
crops.vols = cell(1, 4);
crops.origins = zeros(4, 2);
crops.nT = numel(frameIdx);
for v = 1:4
    y0 = boxes(v, 1); y1 = boxes(v, 2); x0 = boxes(v, 3); x1 = boxes(v, 4);
    yA = y0 + 1; yB = y1; xA = x0 + 1; xB = x1;
    bh = max(yB - yA + 1, 1);
    bw = max(xB - xA + 1, 1);
    vol = zeros(bh, bw, numel(frameIdx), 'single');
    for t = 1:numel(frameIdx)
        fr = read_bin_frame(mems{v}, frameIdx(t));
        vol(:, :, t) = single(fr(yA:yB, xA:xB));
    end
    crops.vols{v} = vol;
    crops.origins(v, :) = [y0, x0];
end
end
