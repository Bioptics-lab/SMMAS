function [entries, centers, primaries] = load_method1(cfg)
%LOAD_METHOD1 Kept method1 neurons from localization.csv + neuron_dictionary.mat.
if nargin < 1, cfg = dataset_config(); end
csvPath = fullfile(cfg.outDir, 'localization.csv');
dictPath = fullfile(cfg.outDir, 'neuron_dictionary.mat');
if ~exist(csvPath, 'file')
    error('missing %s', csvPath);
end
rows = read_csv(csvPath);
if exist(dictPath, 'file')
    d = load(dictPath);
    dictionary = d.merged;
else
    dictionary = [];
end
n = numel(rows);
entries = [];
centers = zeros(n, 3);
primaries = zeros(n, 1);
for i = 1:n
    idx = csv_field(rows(i), 'idx', i - 1);
    idx = double(idx) + 1;
    col = csv_field(rows(i), 'col', NaN);
    rowY = csv_field(rows(i), 'row', NaN);
    z = csv_field(rows(i), 'z_um', NaN);
    pv = csv_field(rows(i), 'primary_view', 1);
    if ~isempty(dictionary)
        di = min(max(idx, 1), numel(dictionary));
        e = dictionary(di);
    else
        e = struct('center', [col, rowY, z], 'center_allview', zeros(2, 4), ...
            'pixels1', [], 'pixels2', [], 'trace', [], 'primary_view', pv, 'peak_corr', NaN);
    end
    e = sync_entry(e);
    e.center = [col, rowY, z];
    e.primary_view = pv;
    if isempty(entries)
        entries = e;
    else
        entries(i) = e;
    end
    centers(i, :) = [col, rowY, z];
    primaries(i) = pv;
end
fprintf('method1: %d kept neurons\n', n);
end

function e = sync_entry(e)
need = {'center', 'center_allview', 'pixels1', 'pixels2', 'trace', 'primary_view', 'peak_corr'};
for i = 1:numel(need)
    if ~isfield(e, need{i})
        if strcmp(need{i}, 'center_allview')
            e.center_allview = zeros(2, 4);
        elseif strcmp(need{i}, 'primary_view')
            e.primary_view = 1;
        elseif strcmp(need{i}, 'peak_corr')
            e.peak_corr = NaN;
        elseif strcmp(need{i}, 'center')
            e.center = [0, 0, 0];
        else
            e.(need{i}) = [];
        end
    end
end
end
