function r = ncc_images(a, b)
a = double(a(:));
b = double(b(:));
a = a - mean(a);
b = b - mean(b);
den = norm(a) * norm(b);
if den < 1e-12
    r = -1;
else
    r = dot(a, b) / den;
end
end
