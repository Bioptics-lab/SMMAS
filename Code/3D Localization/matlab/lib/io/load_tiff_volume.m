function vol = load_tiff_volume(path)
%LOAD_TIFF_VOLUME Load a 3D TIFF as double (Y, X, Z).
info = imfinfo(path);
n = numel(info);
img0 = im2double(imread(path, 1, 'Info', info));
if ndims(img0) > 2
    img0 = img0(:, :, 1);
end
vol = zeros(size(img0, 1), size(img0, 2), n);
vol(:, :, 1) = img0;
for k = 2:n
    frame = im2double(imread(path, k, 'Info', info));
    if ndims(frame) > 2
        frame = frame(:, :, 1);
    end
    vol(:, :, k) = frame;
end
if size(vol, 1) < size(vol, 2) && size(vol, 1) < size(vol, 3)
    vol = permute(vol, [2, 3, 1]);
end
end
