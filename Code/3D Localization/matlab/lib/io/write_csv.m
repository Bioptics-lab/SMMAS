function write_csv(path, rows, fields)
%WRITE_CSV Write a struct array as CSV (empty rows still get a header).
if nargin < 3 || isempty(fields)
    if isempty(rows)
        error('fields required when rows is empty');
    end
    fields = fieldnames(rows);
end
n = numel(rows);
nF = numel(fields);
C = cell(max(n, 0) + 1, nF);
C(1, :) = fields(:)';
for i = 1:n
    for j = 1:nF
        C{i + 1, j} = csv_cell(rows(i).(fields{j}));
    end
end
if n == 0
    C = fields(:)';
end
writecell(C, path);
end

function s = csv_cell(v)
if isempty(v) && ~ischar(v) && ~isstring(v)
    s = '';
elseif isnumeric(v) && isscalar(v)
    if ~isfinite(v)
        s = '';
    else
        s = v;
    end
elseif islogical(v) && isscalar(v)
    s = double(v);
elseif ischar(v) || isstring(v)
    s = char(v);
else
    s = char(string(v));
end
end
