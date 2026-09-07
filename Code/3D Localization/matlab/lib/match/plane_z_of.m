function z = plane_z_of(planeId, cfg)
if nargin < 2, cfg = dataset_config(); end
z = cfg.planeZBase(planeId);
flipped = logical(cfg.planeZFlipped);
side = fullfile(cfg.outDir, 'match', 'plane_z_flipped.json');
if exist(side, 'file')
    j = jsondecode(fileread(side));
    if isfield(j, 'flipped')
        flipped = logical(j.flipped);
    end
end
if flipped
    z = -z;
end
end
