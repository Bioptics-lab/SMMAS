function s2p = load_suite2p(planeDir)
%LOAD_SUITE2P Load suite2p plane0 using numeric npy + s2p_matlab sidecars.
metaDir = fullfile(planeDir, 's2p_matlab');
opsFile = fullfile(metaDir, 'ops.json');
if ~exist(opsFile, 'file')
    error('missing %s — run matlab/export_s2p_meta.py once', opsFile);
end
opsJ = jsondecode(fileread(opsFile));
xpixCat = readNPY(fullfile(metaDir, 'xpix.npy'));
ypixCat = readNPY(fullfile(metaDir, 'ypix.npy'));
lamCat = readNPY(fullfile(metaDir, 'lam.npy'));
off = double(readNPY(fullfile(metaDir, 'pix_offset.npy')));
cnt = double(readNPY(fullfile(metaDir, 'pix_count.npy')));
med = readNPY(fullfile(metaDir, 'med.npy'));
n = numel(cnt);
stat = struct('xpix', cell(n, 1), 'ypix', [], 'lam', [], 'med', []);
for i = 1:n
    a = off(i) + 1;
    b = off(i) + cnt(i);
    if cnt(i) < 1
        stat(i).xpix = zeros(0, 1);
        stat(i).ypix = zeros(0, 1);
        stat(i).lam = zeros(0, 1);
    else
        stat(i).xpix = double(xpixCat(a:b));
        stat(i).ypix = double(ypixCat(a:b));
        stat(i).lam = double(lamCat(a:b));
    end
    if size(med, 1) >= i
        stat(i).med = double(med(i, :));
    else
        stat(i).med = [mean(stat(i).xpix), mean(stat(i).ypix)];
    end
end
iscell = double(readNPY(fullfile(planeDir, 'iscell.npy')));
if size(iscell, 2) >= 2
    cellMask = iscell(:, 1) > 0.5;
else
    cellMask = iscell(:) > 0.5;
end
s2p.stat = stat;
s2p.F = double(readNPY(fullfile(planeDir, 'F.npy')));
s2p.Fneu = double(readNPY(fullfile(planeDir, 'Fneu.npy')));
s2p.iscell = cellMask;
s2p.Order_neuron = find(cellMask);
s2p.ops.meanImg = double(readNPY(fullfile(metaDir, 'meanImg.npy')));
s2p.ops.Ly = double(opsJ.Ly);
s2p.ops.Lx = double(opsJ.Lx);
s2p.ops.nframes = double(opsJ.nframes);
yoffPath = fullfile(metaDir, 'yoff.npy');
xoffPath = fullfile(metaDir, 'xoff.npy');
if exist(yoffPath, 'file')
    s2p.ops.yoff = double(readNPY(yoffPath));
else
    s2p.ops.yoff = [];
end
if exist(xoffPath, 'file')
    s2p.ops.xoff = double(readNPY(xoffPath));
else
    s2p.ops.xoff = [];
end
s2p.planeDir = planeDir;
end
