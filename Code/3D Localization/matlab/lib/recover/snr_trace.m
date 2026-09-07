function s = snr_trace(tr)
tr = double(tr(:));
if numel(tr) < 10
    s = 0;
    return
end
med = median(tr);
sig = prctile(tr, 95) - med;
d = abs(diff(tr));
mad = median(d);
if mad > 1e-12
    noise = mad / 0.6745;
else
    noise = std(tr) + 1e-12;
end
s = sig / (noise + 1e-12);
end
