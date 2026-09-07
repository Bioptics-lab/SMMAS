function r = corrcoef_scalar(a, b)
a = double(a(:));
b = double(b(:));
if numel(a) < 2 || numel(b) < 2 || numel(a) ~= numel(b)
    r = NaN;
    return
end
if std(a) == 0 || std(b) == 0
    r = NaN;
    return
end
C = corrcoef(a, b);
r = C(1, 2);
end
