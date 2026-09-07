function movie = load_movie_from_tiff(tiffPath, ly, lx, tStride, yoff, xoff)
%LOAD_MOVIE_FROM_TIFF Block-average TIFF pages to (Y, X, nAvg) single.
if nargin < 5, yoff = []; end
if nargin < 6, xoff = []; end
info = imfinfo(tiffPath);
nOk = numel(info);
tStride = max(int32(tStride), 1);
nAvg = floor(nOk / double(tStride));
if nAvg < 1
    error('TIFF %s has %d pages < t_stride=%d', tiffPath, nOk, tStride);
end
fprintf('  TIFF fallback %s: average %d/%d pages -> %d frames (bin=%d)\n', ...
    tiffPath, nAvg * double(tStride), nOk, nAvg, tStride);
movie = zeros(ly, lx, nAvg, 'single');
yoff = yoff(:);
xoff = xoff(:);
for i = 1:nAvg
    acc = zeros(ly, lx);
    for j = 1:double(tStride)
        idx = (i - 1) * double(tStride) + j;
        frame = double(imread(tiffPath, idx, 'Info', info));
        if ndims(frame) > 2
            frame = frame(:, :, 1);
        end
        if ~isequal(size(frame), [ly, lx])
            error('TIFF page %d shape mismatch vs ops %dx%d', idx, ly, lx);
        end
        if ~isempty(yoff) && ~isempty(xoff) && idx <= numel(yoff)
            dy = round(yoff(idx));
            dx = round(xoff(idx));
            if dy ~= 0 || dx ~= 0
                frame = circshift(frame, [-dy, -dx]);
            end
        end
        acc = acc + frame;
    end
    movie(:, :, i) = single(acc / double(tStride));
end
end
