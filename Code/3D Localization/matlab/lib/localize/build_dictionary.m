function dict = build_dictionary(neuronCenteri, rawtrace, centersAllview, pixels, cori)
%BUILD_DICTIONARY Per-neuron struct array from estimate_depth outputs.
n = size(neuronCenteri, 1);
dict = struct('center', {}, 'trace', {}, 'center_allview', {}, ...
    'pixels1', {}, 'pixels2', {}, 'cori_allneuron_allz', {}, 'peak_corr', {}, 'primary_view', {});
for ii = 1:n
    dict(ii).center = neuronCenteri(ii, :);
    dict(ii).trace = rawtrace(ii, :);
    dict(ii).center_allview = squeeze(centersAllview(:, ii, :));
    dict(ii).pixels1 = pixels(ii).neuron_pixels_delta1(:);
    dict(ii).pixels2 = pixels(ii).neuron_pixels_delta2(:);
    if ndims(cori) == 2
        dict(ii).cori_allneuron_allz = cori(:, ii);
        dict(ii).peak_corr = max(cori(:, ii));
    else
        dict(ii).cori_allneuron_allz = cori(:);
        dict(ii).peak_corr = max(cori(:));
    end
    dict(ii).primary_view = 1;
end
end
