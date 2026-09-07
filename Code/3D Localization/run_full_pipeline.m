% Thin launcher: add matlab/ and matlab/lib/* to the path, then run the pipeline.
% cd into matlab/ so this script name does not shadow matlab/run_full_pipeline.m.
root = fileparts(mfilename('fullpath'));
matlabDir = fullfile(root, 'matlab');
addpath(matlabDir);
addpath(fullfile(matlabDir, 'lib', 'io'));
addpath(fullfile(matlabDir, 'lib', 'psf'));
addpath(fullfile(matlabDir, 'lib', 'localize'));
addpath(fullfile(matlabDir, 'lib', 'match'));
addpath(fullfile(matlabDir, 'lib', 'recover'));
here = pwd;
cd(matlabDir);
try
    run_full_pipeline();
catch err
    cd(here);
    rethrow(err);
end
cd(here);
