function [trusted, weak] = classify_matches(matches, cfg)
if nargin < 2, cfg = dataset_config(); end
trusted = matches([]);
weak = matches([]);
if isempty(matches)
    return
end
for i = 1:numel(matches)
    ok = csv_field(matches(i), 'corr', -inf) >= cfg.trustRMin ...
        && csv_field(matches(i), 'd_xy_px', inf) <= cfg.trustDXyPx ...
        && csv_field(matches(i), 'd_z_um', inf) <= cfg.trustDZUm;
    if ok
        trusted = [trusted; matches(i)]; %#ok<AGROW>
    else
        weak = [weak; matches(i)]; %#ok<AGROW>
    end
end
end
