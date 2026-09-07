function merged = merge_dictionary1(d1, d2, d3, d4, distThr, corrThr)
%MERGE_DICTIONARY1 Merge four view dictionaries by scaled 3D distance + trace corr.
if nargin < 5, distThr = 40; end
if nargin < 6, corrThr = 0.5; end
d1 = d1(:); d2 = d2(:); d3 = d3(:); d4 = d4(:);
same21 = find_same(d2, d1, distThr, corrThr);
same31 = find_same(d3, d1, distThr, corrThr);
same41 = find_same(d4, d1, distThr, corrThr);
d1 = average_matched(d1, {d2, d3, d4}, {same21, same31, same41});
d2 = drop_indices(d2, same21(:, 1));
d3 = drop_indices(d3, same31(:, 1));
d4 = drop_indices(d4, same41(:, 1));
same32 = find_same(d3, d2, distThr, corrThr);
same42 = find_same(d4, d2, distThr, corrThr);
d2 = average_matched(d2, {d3, d4}, {same32, same42});
merged = [d1; d2];
d3 = drop_indices(d3, same32(:, 1));
d4 = drop_indices(d4, same42(:, 1));
same43 = find_same(d4, d3, distThr, corrThr);
d3 = average_matched(d3, {d4}, {same43});
merged = [merged; d3];
d4 = drop_indices(d4, same43(:, 1));
merged = [merged; d4];
end

function pairs = find_same(dictA, dictB, distThr, corrThr)
pairs = zeros(0, 2);
for ia = 1:numel(dictA)
    for ib = 1:numel(dictB)
        if scaled_distance(dictA(ia).center, dictB(ib).center) >= distThr
            continue
        end
        if corrcoef_scalar(dictA(ia).trace, dictB(ib).trace) > corrThr
            pairs(end+1, :) = [ia, ib]; %#ok<AGROW>
        end
    end
end
end

function d = scaled_distance(c1, c2)
adis = double(c1(1:3)) - double(c2(1:3));
adis(1:2) = adis(1:2) * 1.04;
adis(3) = adis(3) * 0.27;
d = sqrt(sum(adis.^2));
end

function base = average_matched(base, others, sameTables)
matched = [];
for k = 1:numel(sameTables)
    tab = sameTables{k};
    if ~isempty(tab)
        matched = [matched; tab(:, 2)]; %#ok<AGROW>
    end
end
matched = unique(matched);
for ib = matched(:)'
    positions = double(base(ib).center(1:3));
    cavs = {double(base(ib).center_allview)};
    for k = 1:numel(sameTables)
        tab = sameTables{k};
        if isempty(tab), continue; end
        hits = find(tab(:, 2) == ib, 1);
        if isempty(hits), continue; end
        ia = tab(hits, 1);
        od = others{k};
        positions = [positions; double(od(ia).center(1:3))]; %#ok<AGROW>
        cavs{end+1} = double(od(ia).center_allview); %#ok<AGROW>
    end
    base(ib).center = mean(positions, 1);
    stacked = zeros(2, 4, numel(cavs));
    for t = 1:numel(cavs)
        c = cavs{t};
        if ndims(c) == 3
            c = squeeze(c);
        end
        stacked(:, :, t) = reshape(c, 2, 4);
    end
    base(ib).center_allview = mean(stacked, 3);
end
end

function d = drop_indices(d, drop)
if isempty(drop)
    return
end
keep = true(numel(d), 1);
drop = drop(drop >= 1 & drop <= numel(d));
keep(drop) = false;
d = d(keep);
end
