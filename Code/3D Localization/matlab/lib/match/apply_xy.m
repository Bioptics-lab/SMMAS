function [x2, y2] = apply_xy(col, row, tfm)
s = tfm.s;
x2 = s .* col + tfm.tx;
y2 = s .* row + tfm.ty;
end
