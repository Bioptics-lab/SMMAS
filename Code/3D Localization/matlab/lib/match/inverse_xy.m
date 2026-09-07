function [col, row] = inverse_xy(x2, y2, tfm)
s = tfm.s;
col = (x2 - tfm.tx) / s;
row = (y2 - tfm.ty) / s;
end
