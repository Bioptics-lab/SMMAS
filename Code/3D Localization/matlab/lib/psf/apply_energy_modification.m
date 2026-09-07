function [balanced, xpsf] = apply_energy_modification(psfList)
%APPLY_ENERGY_MODIFICATION Compensate Z-dependent PSF energy decay.
n = numel(psfList);
curves = cell(n, 1);
xpsf = 0;
for i = 1:n
    curves{i} = squeeze(sum(sum(psfList{i}, 1), 2));
    xpsf = xpsf + curves{i}(:);
end
xpsf = xpsf(:);
w = 1 ./ max(xpsf, realmin('double'));
w = w / max(w);
balanced = cell(n, 1);
for i = 1:n
    p = psfList{i};
    balanced{i} = p .* reshape(w, 1, 1, []);
end
end
