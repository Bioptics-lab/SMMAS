function rows = read_csv(path)
%READ_CSV Load a CSV as a struct array (numeric fields converted when possible).
if ~exist(path, 'file')
    rows = struct([]);
    return
end
T = readtable(path, 'TextType', 'char', 'Delimiter', ',');
if height(T) == 0
    rows = table2struct(T);
    return
end
rows = table2struct(T);
fn = fieldnames(rows);
for i = 1:numel(rows)
    for k = 1:numel(fn)
        v = rows(i).(fn{k});
        if ischar(v) || isstring(v)
            vs = strtrim(char(v));
            if isempty(vs)
                rows(i).(fn{k}) = '';
            else
                num = str2double(vs);
                if ~isnan(num)
                    rows(i).(fn{k}) = num;
                else
                    rows(i).(fn{k}) = vs;
                end
            end
        end
    end
end
end
