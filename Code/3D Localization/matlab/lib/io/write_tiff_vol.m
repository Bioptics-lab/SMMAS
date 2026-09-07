function write_tiff_vol(path, vol)
%WRITE_TIFF_VOL Write a Y,X,Z float32 multi-page TIFF.
nZ = size(vol, 3);
t = Tiff(path, 'w');
cleanup = onCleanup(@() close(t));
tag.ImageLength = size(vol, 1);
tag.ImageWidth = size(vol, 2);
tag.Photometric = Tiff.Photometric.MinIsBlack;
tag.BitsPerSample = 32;
tag.SamplesPerPixel = 1;
tag.RowsPerStrip = size(vol, 1);
tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tag.SampleFormat = Tiff.SampleFormat.IEEEFP;
tag.Compression = Tiff.Compression.None;
for iz = 1:nZ
    t.setTag(tag);
    t.write(single(vol(:, :, iz)));
    if iz < nZ
        t.writeDirectory();
    end
end
end
