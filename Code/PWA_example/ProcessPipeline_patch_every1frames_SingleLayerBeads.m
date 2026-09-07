%clc; clear; close all;
%% 
% PSF Process

%%%% parameter %%%%
XYcenter = [128,128];
zcenter = 71;
Reconstruct_range = 80;
zrange = (zcenter-Reconstruct_range/2):1:(zcenter+Reconstruct_range/2);
%%%% load psf stack & seg %%%%
rootDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(rootDir, 'data');
path = fullfile(dataDir, 'PSF', 'PSF_stack');
filename = {fullfile(path, 'file_1.tif');
            fullfile(path, 'file_2.tif');
            fullfile(path, 'file_3.tif');
            fullfile(path, 'file_4.tif')};
info_PSF = imfinfo(filename{1});
num_images = numel(info_PSF);
psf_lens=[];
PSF_resizefactor = 1;
for views = 1:4
    psf_view = loadtiff(filename{views});
    %psf_view = psf_view(79:177,78:178,(zcenter-Reconstruct_range/2):(zcenter+Reconstruct_range/2))/30;
    [psf_view_current] = psf_seg_TPLFM(psf_view,XYcenter,zrange);
    psf_lens = cat(4,psf_lens,psf_view_current);
    PSF_projection = max(psf_view_current, [], 3);

    % Display result
    figure;
    %subplot(1,2,1);
    imshow(squeeze(PSF_projection), []);
    title('PSF maximum intensity projection');
end
%%%% psf save %%%%
psfname = fullfile(path, 'PSF_1205.mat');
save(psfname, 'psf_lens');
%% 
% Data & PSF size

PSF_zoom = 15;
Image_zoom = 2;
PSF_pixel = 256;
Image_pixel = 1024;
resizefactor = (PSF_zoom/Image_zoom)*(PSF_pixel/Image_pixel);
%% 
% Block size
block_size = 1;

% Define path and filename prefix
DataPath = fullfile(dataDir, 'ReconstructOrigin');
filename_prefix = 'data_';

% Load raw data
Data_name = {fullfile(DataPath, 'processed_File1.tif');
    fullfile(DataPath, 'processed_File2.tif');
    fullfile(DataPath, 'processed_File3.tif');
    fullfile(DataPath, 'processed_File4.tif')};
info_data = imfinfo(Data_name{1});
num_video = numel(info_data);

% Process and save each data block
for i = 1:ceil(num_video/block_size)
    % Locate start and end of the current block
    start_idx = (i-1)*block_size+1;
    end_idx = min(i*block_size, num_video);
    
    % Read data of the current block
    Data_raw = [];
    for view = 1:4
        for k = start_idx:1:end_idx
            %Img_current = imread(Data_name{view}, k, 'Info', info_data);
            Img_current = loadtiff(Data_name{view});
            Img_current = imresize(Img_current,[resizefactor*info_data(1).Width resizefactor*info_data(1).Height],'cubic');
            Data_raw(:,:,view,(k-start_idx)/1+1) = Img_current;
        end
    end
    
    % Save data of the current block
    filename = [filename_prefix sprintf('%04d', i) '.mat'];
    full_filename = fullfile(DataPath, filename);
    save(full_filename, 'Data_raw', '-v7.3');
end

%%% Data Reconstruction

%%%% Deconcolution %%%%
psf_path = fullfile(path, 'PSF_1205.mat');
savepath = DataPath;
Img_pixel = resizefactor*info_data(1).Width-1;
Psf_pixel = 241/PSF_resizefactor;
iter = 8;
for iii = 1:ceil(num_video/block_size)
    data_path = fullfile(DataPath, [filename_prefix sprintf('%04d', iii) '.mat']);
    ObjRECON_t = FLFM_deconRL_GPU(psf_path,data_path,savepath,Img_pixel,Psf_pixel,iter);
    % Define result filename prefix and save path
    filename_result_prefix = 'data_';
    
    % Loop over slices and save as a multipage TIFF
    for i = 1:size(ObjRECON_t, 4)
        %i = num_save;
        % Build filename for the current slice
        filename = [filename_result_prefix sprintf('%04d', i*1+(iii-1)*block_size) '.tif'];
        full_filename = fullfile(savepath, filename);
        
        % Create a multipage TIFF with the Tiff class
        t = Tiff(full_filename, 'w');
        
        % Write all channels into the multipage TIFF
        for j = 1:size(ObjRECON_t, 3)
            t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
            t.setTag('Compression', Tiff.Compression.None);
            t.setTag('BitsPerSample', 32); % Set to 32-bit floating-point
            t.setTag('SamplesPerPixel', 1);
            t.setTag('SampleFormat', Tiff.SampleFormat.IEEEFP); % Set to IEEE floating-point
            t.setTag('ExtraSamples', Tiff.ExtraSamples.Unspecified);
            
            t.setTag('ImageWidth', size(ObjRECON_t, 2));
            t.setTag('ImageLength', size(ObjRECON_t, 1));
            t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
            
            t.write(ObjRECON_t(:,:,j,i));
            if (j ~= size(ObjRECON_t, 3))
                t.writeDirectory();
            end
        end
        
        % Close the TIFF object
        t.close();
    end
end
