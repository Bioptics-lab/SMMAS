% clear;
% clc;
%
% Manually select view image files

% disp('Select the reference (reconstructed) image file:');
% [view0File, view0Path] = uigetfile('*.tif', 'Select the reference (reconstructed) image');
% view0Image = loadtiff(fullfile(view0Path, view0File));
%
% disp('Select the view 1 image file:');
% [view1File, view1Path] = uigetfile('*.tif', 'Select view 1 image');
% view1Image = loadtiff(fullfile(view1Path, view1File))-32768;
% view1Image(view1Image < 150) = 0;
%
% disp('Select the view 2 image file:');
% [view2File, view2Path] = uigetfile('*.tif', 'Select view 2 image');
% view2Image = loadtiff(fullfile(view2Path, view2File))-32768;
% view2Image(view2Image<100) = 0;
%
% disp('Select the view 3 image file:');
% [view3File, view3Path] = uigetfile('*.tif', 'Select view 3 image');
% view3Image = loadtiff(fullfile(view3Path, view3File))-32768;
%
% disp('Select the view 4 image file:');
% [view4File, view4Path] = uigetfile('*.tif', 'Select view 4 image');
% view4Image = loadtiff(fullfile(view4Path, view4File))-32768;

% Combine images into a multi-view array
views = cat(3, view1Image, view2Image, view3Image, view4Image, view0Image);

% Set thresholds
threshold_value = 15;  % Intensity threshold (adjust as needed)
delta_threshold = 20;  % Clustering threshold (maximum distance between centroids)

% Keep only pixels above the threshold
views(views < threshold_value) = 0;

% Label connected components
L = bwlabel(views(:,:,5));  % Connected-component labeling

% Get connected-component properties
stat = regionprops(L, 'Area', 'Centroid', 'BoundingBox');

% Initialize validRegions
validRegions = [];

% Iterate over each row in stat
for i = 1:height(stat)
    % Get the area of the current region
    area = stat(i, :).Area;  % Ensure this is a scalar

    % Keep the region if its area is within the allowed range
    if area > 20 && area < 1000
        % Keep this region
        validRegions = [validRegions; stat(i, :)];
    end
end

% Print the number of valid regions
disp(['Number of valid regions: ', num2str(height(validRegions))]);

% Initialize an empty centroids matrix
centroids = [];

% Initialize an empty Positions matrix
Positions = [];

% Iterate over validRegions and extract each centroid
for i = 1:height(validRegions)
    % Get the centroid of the current region
    currentCentroid = validRegions(i,:).Centroid;

    % Get x and y of the current region
    x = currentCentroid(1);  % x coordinate
    y = currentCentroid(2);  % y coordinate

    % Check whether the centroid is inside the allowed field of view
    if x > 21 && x < 1004 && y > 21 && y < 1004
        % Append the centroid to Positions
        Positions = [Positions; currentCentroid];
    end
end

% Match positional offsets in the other views
delta_beads = zeros(4, 2, size(Positions, 1)); % (number of views - 1) x offset x number of beads

delta_ini = [30,-2;-2,21;19,-27;-18,-7;0,0];

for viewIdx = 1:4
    for ballIdx = 1:size(Positions, 1)
        ballPos = Positions(ballIdx, :);

        % Matching window sized for a 1024x1024 image
        rowRange = max(1, ballPos(2) - 20 - delta_ini(viewIdx,1)):min(1024, ballPos(2) + 20 - delta_ini(viewIdx,1));
        colRange = max(1, ballPos(1) - 20 - delta_ini(viewIdx,2)):min(1024, ballPos(1) + 20 - delta_ini(viewIdx,2));

        area = views(round(rowRange), round(colRange), viewIdx);
        Lv = bwlabel(area);
        statv = regionprops(Lv, 'Area', 'Centroid', 'BoundingBox');

        if ~isempty(statv)
            % Collect all centroid coordinates
            centroids = vertcat(statv.Centroid);  % Stack all centroids into a matrix

            % Euclidean distance of each centroid to (21, 21)
            distances = vecnorm(centroids - [21, 21], 2, 2);

            % Find the nearest centroid
            minIdx = find(distances == min(distances));

            % Store the offset in delta_beads
            delta_beads(viewIdx, :, ballIdx) = centroids(minIdx, :) - [21, 21];
        end
    end
end

% Fit the offsets
for viewIdx = 1:4
    xOffset = squeeze(delta_beads(viewIdx, 1, :));
    yOffset = squeeze(delta_beads(viewIdx, 2, :));
    %[fitResultX, gofX] = createFitx(Positions(:, 1), Positions(:, 2), xOffset);
    %[fitResultY, gofY] = createFitx(Positions(:, 1), Positions(:, 2), yOffset);
    [fitResultX, gofX] = createFitx(Positions(:, 1), Positions(:, 2), xOffset);
    [fitResultY, gofY] = createFitx(Positions(:, 1), Positions(:, 2), yOffset);

    fitGroupsX{viewIdx} = fitResultX;
    fitGroupsY{viewIdx} = fitResultY;
end

% Example: evaluate the fit at (x_query, y_query)
x_query = 512; % Query location X
y_query = 512; % Query location Y

%%% Fourier-plane coordinates of the beams for each view %%%
fxyPosition = [-0.56/0.8,0;0,-0.56/0.8;0.56/0.8,0;0,0.56/0.8];
fittedPosition = zeros(4,2);

% Evaluate the fit for each view
for viewIdx = 1:4
    % Evaluate fitResult
    fittedPosition(viewIdx,1) = fitGroupsX{viewIdx}(x_query, y_query);
    fittedPosition(viewIdx,2) = fitGroupsY{viewIdx}(x_query, y_query);

    fprintf('View %d: Fitted X = %.2f, Fitted Y = %.2f\n', viewIdx, fittedPosition(viewIdx,1), fittedPosition(viewIdx,2));
end

fitted_center_pos = fittedPosition;

% Fit the offsets
for viewIdx = 1:4
    xOffset = squeeze(delta_beads(viewIdx, 1, :));
    yOffset = squeeze(delta_beads(viewIdx, 2, :));
    %[fitResultX, gofX] = createFitx(Positions(:, 1), Positions(:, 2), xOffset);
    %[fitResultY, gofY] = createFitx(Positions(:, 1), Positions(:, 2), yOffset);
    [fitResultX, gofX] = createFitx(Positions(:, 1), Positions(:, 2), xOffset);
    [fitResultY, gofY] = createFitx(Positions(:, 1), Positions(:, 2), yOffset);

    fitGroupsX{viewIdx} = fitResultX;
    fitGroupsY{viewIdx} = fitResultY;

    % Read the original image
    originalImage = views(:,:,viewIdx);
    [imageHeight, imageWidth] = size(originalImage);

    % Generate the grid
    [originalX, originalY] = meshgrid(1:imageWidth, 1:imageHeight);

    % Displacement from the fitted surface
    displacementX = fitResultX(originalX, originalY)-fitted_center_pos(viewIdx,1);
    displacementY = fitResultY(originalX, originalY)-fitted_center_pos(viewIdx,2);

    % New grid locations
    newX = originalX + displacementX;
    newY = originalY + displacementY;

    % Warp the image by interpolation
    transformedImage = interp2(originalX, originalY, double(originalImage), newX, newY, 'linear', NaN);

    % Display results
    figure;
    subplot(1, 2, 1);
    imshow(originalImage, []);
    title('Original Image');

    subplot(1, 2, 2);
    imshow(transformedImage, []);
    title('Transformed Image');

    %saveastiff(transformedImage,['PWA' num2str(viewIdx) '.tif']);

    originalImage = views(:,:,viewIdx);
    % Translation
    dx = -fitted_center_pos(viewIdx,1)+delta_ini(viewIdx,2); 
    dy = -fitted_center_pos(viewIdx,2)+delta_ini(viewIdx,1);
    translatedMatrix = imtranslate(originalImage, [dx, dy], 'FillValues', 0);
    transformedImage(isnan(transformedImage)) = 0;
    translatedMatrixPWA = imtranslate(transformedImage, [dx, dy], 'FillValues', 0);

    % Save as TIF files
    outputFile1 = ['translatedMatrix' num2str(viewIdx) '.tif'];
    saveastiff(translatedMatrix,outputFile1);
    outputFile2 = ['translatedMatrixPWA' num2str(viewIdx) '.tif'];
    saveastiff(translatedMatrixPWA,outputFile2);

end
