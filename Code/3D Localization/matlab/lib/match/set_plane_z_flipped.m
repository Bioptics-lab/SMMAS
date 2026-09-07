function set_plane_z_flipped(flipped, cfg)
if nargin < 2, cfg = dataset_config(); end
ensure_dir(fullfile(cfg.outDir, 'match'));
write_json(fullfile(cfg.outDir, 'match', 'plane_z_flipped.json'), struct('flipped', logical(flipped)));
fprintf('PLANE_Z_FLIPPED -> %d\n', logical(flipped));
end
