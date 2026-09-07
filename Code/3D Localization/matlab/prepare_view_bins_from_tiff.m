function prepare_view_bins_from_tiff(cfg)
%PREPARE_VIEW_BINS_FROM_TIFF Stream Views/{v}/file_{v}.tif -> suite2p/plane0/data.bin.
if nargin < 1, cfg = dataset_config(); end
fprintf('DATA_ROOT = %s\n', cfg.dataRoot);
for v = 1:4
    convert_one(v, cfg);
end
fprintf('Done. View data.bin check complete.\n');
end

function convert_one(v, cfg)
plane = view_plane_dir(v, cfg);
s2p = load_suite2p(plane);
ly = s2p.ops.Ly;
lx = s2p.ops.Lx;
nf = s2p.ops.nframes;
outBin = fullfile(plane, 'data.bin');
if bin_is_valid(outBin, nf, ly, lx)
    info = dir(outBin);
    fprintf('view%d: skip existing %s (%d bytes)\n', v, outBin, info.bytes);
    return
end
d = fileparts(fileparts(plane));
tiffPath = fullfile(d, sprintf('file_%d.tif', v));
if ~exist(tiffPath, 'file')
    fprintf('view%d: no TIFF and no valid data.bin (skipped)\n', v);
    return
end
info = imfinfo(tiffPath);
if numel(info) < nf
    error('TIFF pages %d < nframes %d', numel(info), nf);
end
yoff = s2p.ops.yoff(:);
xoff = s2p.ops.xoff(:);
fprintf('view%d: streaming %s -> %s  (%d x %d x %d)\n', v, tiffPath, outBin, nf, ly, lx);
tmp = [outBin, '.tmp'];
fid = fopen(tmp, 'wb');
if fid < 0
    error('cannot write %s', tmp);
end
nNz = 0;
try
    for i = 1:nf
        frame = imread(tiffPath, i, 'Info', info);
        if ndims(frame) > 2
            frame = frame(:, :, 1);
        end
        if ~isequal(size(frame), [ly, lx])
            error('TIFF page %d shape mismatch', i);
        end
        if ~isempty(yoff) && ~isempty(xoff) && i <= numel(yoff)
            dy = round(yoff(i));
            dx = round(xoff(i));
            if dy ~= 0 || dx ~= 0
                frame = circshift(frame, [-dy, -dx]);
            end
        end
        clipped = frame_to_int16(frame);
        if any(clipped(:) ~= 0)
            nNz = nNz + 1;
        end
        fwrite(fid, clipped', 'int16'); % C-order X-fastest
        if mod(i, 500) == 0
            fprintf('    page %d/%d\n', i, nf);
        end
    end
catch err
    fclose(fid);
    if exist(tmp, 'file'), delete(tmp); end
    rethrow(err);
end
fclose(fid);
expect = nf * ly * lx * 2;
got = dir(tmp);
if got.bytes ~= expect
    delete(tmp);
    error('temp data.bin size %d != expect %d', got.bytes, expect);
end
if exist(outBin, 'file')
    delete(outBin);
end
movefile(tmp, outBin);
if nNz == 0
    error('view %d converted movie is all zeros', v);
end
fprintf('  wrote %s  shape=(%d,%d,%d) dtype=int16  nonzero_frames=%d/%d\n', ...
    outBin, nf, ly, lx, nNz, nf);
end

function clipped = frame_to_int16(frame)
if isa(frame, 'uint16')
    clipped = int16(min(max(int32(frame), 0), 32767));
else
    clipped = int16(min(max(round(double(frame)), -32768), 32767));
end
end
