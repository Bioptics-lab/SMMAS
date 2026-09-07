function ab = ab_test_from_depths(results, trusted, c1Orig, cfg)
if nargin < 4, cfg = dataset_config(); end
A = score_flip(false, results, trusted, c1Orig, cfg);
B = score_flip(true, results, trusted, c1Orig, cfg);
if B.score < A.score
    choose = 'B';
else
    choose = 'A';
end
ab.A = A;
ab.B = B;
ab.choose = choose;
end

function out = score_flip(flipped, results, trusted, c1Orig, cfg)
plane = zeros(1, numel(cfg.planeIds));
for i = 1:numel(cfg.planeIds)
    z = cfg.planeZBase(cfg.planeIds(i));
    if flipped, z = -z; end
    plane(i) = z;
end
vals = [];
out = struct();
for i = 1:numel(cfg.planeIds)
    pid = cfg.planeIds(i);
    zs = [];
    for r = 1:numel(results)
        if results(r).m2.plane_id == pid
            zs(end+1) = results(r).best_z; %#ok<AGROW>
        end
    end
    if isempty(zs)
        recMed = nan;
    else
        recMed = median(abs(zs - plane(i)));
    end
    zsT = [];
    for t = 1:numel(trusted)
        if trusted(t).m2_plane ~= pid, continue; end
        i1 = trusted(t).m1_idx + 1;
        if i1 >= 1 && i1 <= size(c1Orig, 1)
            zsT(end+1) = c1Orig(i1, 3); %#ok<AGROW>
        end
    end
    if isempty(zsT)
        trustMed = nan;
    else
        trustMed = median(abs(zsT - plane(i)));
    end
    out.(sprintf('rec_med_abs_%d', pid)) = recMed;
    out.(sprintf('trust_med_abs_%d', pid)) = trustMed;
    if isempty(zs)
        out.(sprintf('rec_median_z_%d', pid)) = nan;
    else
        out.(sprintf('rec_median_z_%d', pid)) = median(zs);
    end
    vals = [vals, recMed, trustMed]; %#ok<AGROW>
end
out.score = mean(vals, 'omitnan');
end
