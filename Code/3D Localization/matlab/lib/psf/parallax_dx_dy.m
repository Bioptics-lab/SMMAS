function [dx, dy] = parallax_dx_dy(nz, viewIdx0, coefXz, coefYz, factor, x0, y0)
%PARALLAX_DX_DY Independent x(z), y(z) lateral offset. viewIdx0 is 0-based.
if nargin < 5, factor = 5.0; end
if nargin < 6, x0 = 60.0; end
if nargin < 7, y0 = 60.0; end
v = viewIdx0 + 1;
dx = coefXz(1, v) / factor * nz + coefXz(2, v) / factor - x0 / factor;
dy = coefYz(1, v) / factor * nz + coefYz(2, v) / factor - y0 / factor;
end
