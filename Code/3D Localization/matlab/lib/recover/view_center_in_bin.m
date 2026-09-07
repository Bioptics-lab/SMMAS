function [cx, cy] = view_center_in_bin(col, row, z, primaryV0, viewV0, moveI, coefXz, coefYz, factor, x0, y0)
[dxP, dyP] = parallax_dx_dy(z, primaryV0, coefXz, coefYz, factor, x0, y0);
[dxV, dyV] = parallax_dx_dy(z, viewV0, coefXz, coefYz, factor, x0, y0);
cx = col + (dxV - dxP) + double(moveI(viewV0 + 1, 2));
cy = row + (dyV - dyP) + double(moveI(viewV0 + 1, 1));
end
