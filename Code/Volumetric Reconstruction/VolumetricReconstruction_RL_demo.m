%%
% PSF Process

%%%% parameter %%%%
XYcenter = [136,134];
zcenter = 26;
Reconstruct_range = 40;
zrange = (zcenter-Reconstruct_range/2):1:(zcenter+Reconstruct_range/2);
%%%% load psf stack & seg %%%%
path = 'PSF_stack';
filename = [path '/file_1.tif'; 
            path '/file_2.tif'; 
            path '/file_3.tif'; 
            path '/file_4.tif'];
info_PSF = imfinfo(filename(1,:));
num_images = numel(info_PSF);
psf_lens=[];
for views = 1:4
    psf_view = loadtiff(filename(views,:));
    [psf_view_current] = psf_seg_TPLFM(psf_view,XYcenter,zrange);
    psf_lens = cat(4,psf_lens,psf_view_current);
end
%%%% psf save %%%%
psfname = [path '\PSF_0301.mat'];
save(psfname, 'psf_lens');
%%
% Data & PSF size

PSF_zoom = 15;
Image_zoom = 5;
PSF_pixel = 256;
Image_pixel = 256;
resizefactor = (PSF_zoom/Image_zoom)*(PSF_pixel/Image_pixel);
%%
% Block size
block_size = 10;

% Path and filename prefix
DataPath = 'Data';
filename_prefix = 'data_block_interval_1_';

% Load raw data
Data_name = [DataPath '/file_1_1.tif';
    DataPath '/file_2_1.tif';
    DataPath '/file_3_1.tif';
    DataPath '/file_4_1.tif'];
info_data = imfinfo(Data_name(1,:));
num_video = numel(info_data);

% Process each block and save
for i = 1:ceil(num_video/block_size)
    % Start and end indices of the current block
    start_idx = (i-1)*block_size+1;
    end_idx = min(i*block_size, num_video);
    
    % Read data of the current block
    Data_raw = [];
    for view = 1:4
        for k = start_idx:1:end_idx
            Img_current = imread(Data_name(view,:), k, 'Info', info_data);
            Img_current = imresize(Img_current,[resizefactor*info_data(1).Width resizefactor*info_data(1).Height],'cubic');
            Data_raw(:,:,view,(k-start_idx)/1+1) = Img_current;
        end
    end
    
    % Save the current block
    filename = [filename_prefix sprintf('%04d', i) '.mat'];
    full_filename = fullfile(DataPath, filename);
    save(full_filename, 'Data_raw', '-v7.3');
end

%%% Data Reconstruction

%%%% Deconvolution %%%%
psf_path = 'PSF_stack/PSF_0301.mat';
savepath = fullfile('Data', 'Reconstruction');
if ~exist(savepath, 'dir')
    mkdir(savepath);
end
Img_pixel = resizefactor*info_data(1).Width-1;
PSF_resizefactor = 1;
Psf_pixel = 141/PSF_resizefactor;
iter = 10;
for iii = 1:ceil(num_video/block_size)
    data_path = ['Data/' filename_prefix sprintf('%04d', iii) '.mat'];
    ObjRECON_t = FLFM_deconRL_GPU(psf_path,data_path,savepath,Img_pixel,Psf_pixel,iter);
    % Result filename prefix
    filename_result_prefix = 'data_';
    
    % Save each time point as a multi-page TIFF
    for i = 1:size(ObjRECON_t, 4)
        filename = [filename_result_prefix sprintf('%04d', i*1+(iii-1)*block_size) '.tif'];
        
        % tifflib may fail if the directory does not exist, the path is
        % relative, or it contains spaces. Switch into the save folder and
        % open the file by name only.
        here = pwd;
        cd(savepath);
        try
            t = Tiff(filename, 'w');
            
            % Write all z-slices into a multi-page TIFF
            for j = 1:size(ObjRECON_t, 3)
                t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
                t.setTag('Compression', Tiff.Compression.None);
                t.setTag('BitsPerSample', 32); % 32-bit floating point
                t.setTag('SamplesPerPixel', 1);
                t.setTag('SampleFormat', Tiff.SampleFormat.IEEEFP); % IEEE floating point
                t.setTag('ExtraSamples', Tiff.ExtraSamples.Unspecified);
                
                t.setTag('ImageWidth', size(ObjRECON_t, 2));
                t.setTag('ImageLength', size(ObjRECON_t, 1));
                t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
                
                t.write(single(ObjRECON_t(:,:,j,i)));
                if (j ~= size(ObjRECON_t, 3))
                    t.writeDirectory();
                end
            end
            
            t.close();
        catch ME
            if exist('t', 'var')
                try, t.close(); end
            end
            cd(here);
            rethrow(ME);
        end
        cd(here);
    end
end
