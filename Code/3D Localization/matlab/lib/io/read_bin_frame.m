function frame = read_bin_frame(mm, t)
%READ_BIN_FRAME Return one YxX frame (1-based t) from a C-order data.bin memmap.
vol = mm.map.Data.vol(:, :, t);
frame = double(vol'); % X,Y -> Y,X
end
