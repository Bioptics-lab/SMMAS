function oimg = loadtiff(path)
%LOADTIFF  Load a multi-page TIFF stack without using Tiff.nextDirectory.
%   The File Exchange loadtiff walks pages with lastDirectory/nextDirectory.
%   That pair often fails on stacks whose IFDs sit after each plane
%   ("Unable to read the next directory"). This reader uses imfinfo + imread.

if ~exist(path, 'file')
    error('loadtiff:FileNotFound', 'File not found: %s', path);
end

info = imfinfo(path);
nPage = numel(info);
sample = imread(path, 1, 'Info', info);
sz = size(sample);
oimg = zeros([sz(1), sz(2), nPage], class(sample));
oimg(:, :, 1) = sample;

for k = 2:nPage
    try
        oimg(:, :, k) = imread(path, k, 'Info', info);
    catch
        warning('loadtiff:TruncatedStack', ...
            'Stopped at page %d of %s (later IFDs are unreadable).', k - 1, path);
        oimg = oimg(:, :, 1:(k - 1));
        break
    end
end
end
