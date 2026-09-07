function v = csv_field(row, name, defaultVal)
%CSV_FIELD Read a struct field, returning defaultVal when missing/empty.
if nargin < 3
    defaultVal = [];
end
if ~isfield(row, name)
    v = defaultVal;
    return
end
v = row.(name);
if isempty(v) && ~isnumeric(v)
    v = defaultVal;
end
end
