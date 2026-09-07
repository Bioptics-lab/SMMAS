function move = compute_move(centers, factor)
%COMPUTE_MOVE Inter-view alignment offsets from PSF centers (1-based row,col).
if nargin < 2, factor = 5.0; end
c = double(centers);
c = c - mean(c, 1);
move = round(c / factor);
end
