function writeNPY(data, filename)
%WRITENPY Write a numeric array as little-endian .npy (C-order).
data = squeeze(data);
if islogical(data)
    data = uint8(data);
end
[descr, prec] = npy_descr(data);
shp = size(data);
if isvector(data) && size(data, 1) == 1 && numel(shp) == 2
    shp = [1, numel(data)];
end
if iscolumn(data) && numel(data) > 1
    shp = [numel(data), 1];
end
shapeStr = sprintf('%d, ', shp);
shapeStr = ['(', shapeStr(1:end-2), ')'];
if numel(shp) == 1
    shapeStr = sprintf('(%d,)', shp);
end
header = sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shapeStr);
header = [char(header), ' '];
% npy 1.0: 6 magic + 2 version + 2 headerlen + header, padded to 16-byte multiple
base = 10;
pad = 16 - mod(base + numel(header) + 1, 16);
if pad == 16
    pad = 0;
end
header = [header, repmat(' ', 1, pad), newline];
fid = fopen(filename, 'wb');
if fid < 0
    error('cannot write %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, uint8([147 78 85 77 80 89]), 'uint8');
fwrite(fid, uint8([1, 0]), 'uint8');
fwrite(fid, uint16(numel(header)), 'uint16');
fwrite(fid, uint8(header), 'uint8');
% C-order: last MATLAB dim varies fastest after flipping
nDim = ndims(data);
if nDim == 2 && isvector(data)
    fwrite(fid, data(:), prec);
else
    payload = permute(data, nDim:-1:1);
    fwrite(fid, payload(:), prec);
end
end

function [descr, prec] = npy_descr(data)
switch class(data)
    case 'double'
        descr = '<f8'; prec = 'float64';
    case 'single'
        descr = '<f4'; prec = 'float32';
    case 'int64'
        descr = '<i8'; prec = 'int64';
    case 'int32'
        descr = '<i4'; prec = 'int32';
    case 'int16'
        descr = '<i2'; prec = 'int16';
    case 'int8'
        descr = '<i1'; prec = 'int8';
    case 'uint64'
        descr = '<u8'; prec = 'uint64';
    case 'uint32'
        descr = '<u4'; prec = 'uint32';
    case 'uint16'
        descr = '<u2'; prec = 'uint16';
    case 'uint8'
        descr = '|u1'; prec = 'uint8';
    otherwise
        error('unsupported class %s', class(data));
end
end
