function pool = load_method2(cfg)
%LOAD_METHOD2 Pool iscell ROIs from recon planes 17/21/25.
if nargin < 1, cfg = dataset_config(); end
pool = struct('plane_id', {}, 'roi_idx', {}, 'z_um', {}, 'x', {}, 'y', {}, ...
    'xpix', {}, 'ypix', {}, 'trace', {}, 'plane_dir', {});
k = 0;
for p = 1:numel(cfg.planeIds)
    pid = cfg.planeIds(p);
    planeDir = plane_suite2p_dir(pid, cfg);
    s2p = load_suite2p(planeDir);
    zUm = plane_z_of(pid, cfg);
    nCell = sum(s2p.iscell(:));
    fprintf('method2 plane%d z=%.1f: FOV %dx%d  iscell=%d/%d\n', ...
        pid, zUm, s2p.ops.Ly, s2p.ops.Lx, nCell, numel(s2p.stat));
    roi = find(s2p.iscell);
    for t = 1:numel(roi)
        ri = roi(t);
        st = s2p.stat(ri);
        xpix = double(st.xpix(:));
        ypix = double(st.ypix(:));
        k = k + 1;
        pool(k).plane_id = pid;
        pool(k).roi_idx = ri - 1; % 0-based like Python
        pool(k).z_um = zUm;
        pool(k).x = mean(xpix);
        pool(k).y = mean(ypix);
        pool(k).xpix = int32(xpix);
        pool(k).ypix = int32(ypix);
        pool(k).trace = s2p.F(ri, :);
        pool(k).plane_dir = planeDir;
    end
end
fprintf('method2: %d iscell ROIs pooled\n', numel(pool));
end
