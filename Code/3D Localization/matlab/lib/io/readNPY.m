function data = readNPY(filename)
%READNPY Load a numeric NumPy .npy array (C or Fortran order, little-endian).
fid = fopen(filename, 'rb');
if fid < 0
    error('cannot open %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
magic = fread(fid, 6, '*uint8')';
if ~isequal(magic, uint8([147 78 85 77 80 89]))
    error('not an npy file: %s', filename);
end
major = fread(fid, 1, '*uint8');
fread(fid, 1, '*uint8'); % minor
if major == 1
    headerLen = double(fread(fid, 1, '*uint16'));
else
    headerLen = double(fread(fid, 1, '*uint32'));
end
header = native2unicode(fread(fid, headerLen, '*uint8')', 'UTF-8');
descr = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens', 'once');
fortran = contains(header, '''fortran_order'': True');
shapeTok = regexp(header, '''shape'':\s*\(([^\)]*)\)', 'tokens', 'once');
if isempty(descr) || isempty(shapeTok)
    error('could not parse npy header: %s', filename);
end
shapeStr = strtrim(shapeTok{1});
if isempty(shapeStr)
    shape = [];
else
    parts = regexp(shapeStr, ',', 'split');
    shape = [];
    for i = 1:numel(parts)
        t = strtrim(parts{i});
        if ~isempty(t)
            shape(end+1) = str2double(t); %#ok<AGROW>
        end
    end
end
dt = descr{1};
[prec, bytes, complexFlag] = npy_dtype(dt);
nElem = prod(shape);
if isempty(shape)
    nElem = 1;
end
if complexFlag
    raw = fread(fid, nElem * 2, prec);
    data = complex(raw(1:2:end), raw(2:2:end));
else
    data = fread(fid, nElem, prec);
end
if isempty(shape)
    return
end
if fortran
    data = reshape(data, shape);
else
    data = reshape(data, fliplr(shape));
    data = permute(data, numel(shape):-1:1);
end
end

function [prec, bytes, complexFlag] = npy_dtype(dt)
complexFlag = false;
if startsWith(dt, {'<', '|', '>'})
    code = dt(2:end);
else
    code = dt;
end
switch code
    case {'f8', 'd'}
        prec = 'float64'; bytes = 8;
    case {'f4', 'f'}
        prec = 'float32'; bytes = 4;
    case {'i8'}
        prec = 'int64'; bytes = 8;
    case {'i4', 'i'}
        prec = 'int32'; bytes = 4;
    case {'i2'}
        prec = 'int16'; bytes = 2;
    case {'i1', 'b'}
        prec = 'int8'; bytes = 1;
    case {'u8'}
        prec = 'uint64'; bytes = 8;
    case {'u4'}
        prec = 'uint32'; bytes = 4;
    case {'u2'}
        prec = 'uint16'; bytes = 2;
    case {'u1', 'B'}
        prec = 'uint8'; bytes = 1;
    case {'c16'}
        prec = 'float64'; bytes = 16; complexFlag = true;
    case {'c8'}
        prec = 'float32'; bytes = 8; complexFlag = true;
    otherwise
        error('unsupported npy dtype %s', dt);
end
end
