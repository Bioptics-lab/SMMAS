function cfg = dataset_config()
%DATASET_CONFIG Paths and optical constants (zoom5 imaging, 256 px, zoom15 PSF).
% Also puts matlab/lib/{io,psf,localize,match,recover} on the MATLAB path.
matlabDir = fileparts(mfilename('fullpath'));
add_matlab_lib(matlabDir);
cfg.matlabDir = matlabDir;
cfg.project = fileparts(matlabDir);
cfg.dataRoot = fullfile(cfg.project, 'Views');
cfg.psfDir = fullfile(cfg.project, 'PSF');
cfg.planesDir = fullfile(cfg.project, 'Planes');
cfg.outDir = fullfile(cfg.project, 'CA1R1f5OutputMatlab');

cfg.factor = 3.0;
cfg.moveDivisor = 3.0;
cfg.psfDzUm = 2.0;
cfg.imageZoom = 5.0;
cfg.psfZoom = 15.0;
cfg.fovPx = 256.0;
cfg.fovUm = 160.0;
cfg.pixelUm = cfg.fovUm / cfg.fovPx;

% 1-based (row, col)
cfg.viewCenters = [138, 121; 138, 123; 143, 124; 145, 123];
cfg.psfZrange = 13:53;
cfg.cropHalfY = 59;
cfg.cropHalfX = 60;

cfg.planeZBase = containers.Map('KeyType', 'double', 'ValueType', 'double');
cfg.planeZBase(14) = -15.0;
cfg.planeZBase(19) = -5.0;
cfg.planeZBase(24) = 5.0;
cfg.planeZBase(29) = 15.0;
cfg.planeIds = [14, 19, 24, 29];
cfg.planeZFlipped = false;
cfg.matchMidPlane = 19;

cfg.tStride = 10;
cfg.traceTStride = 10;
cfg.forceRecalib = false;

cfg.distThr = 40.0;
cfg.corrThr = 0.5;
cfg.depthStepUm = 2;
cfg.filterMinPeakCorr = 0.30;
cfg.filterMaxCenterSpreadPx = 45.0;
cfg.filterMinMeanTraceCorr = 0.10;
cfg.filterMinViews = 2;

cfg.matchMode = 'loose';
cfg.matchGates = struct( ...
    'baseline', struct('dXyPx', 40.0, 'dZUm', 20.0, 'rMin', 0.12), ...
    'loose', struct('dXyPx', 50.0, 'dZUm', 24.0, 'rMin', 0.08), ...
    'zswap', struct('dXyPx', 40.0, 'dZUm', 20.0, 'rMin', 0.12));
cfg.wXy = 1.0;
cfg.wZ = 0.35;
cfg.wR = 20.0;

cfg.trustRMin = 0.40;
cfg.trustDXyPx = 20.0;
cfg.trustDZUm = 12.0;
cfg.zSearchHalf = 20.0;
cfg.zStep = 2.0;
cfg.diskRadius = 5;
cfg.tStrideRecover = 15;
cfg.rBest = 0.22;
cfg.rPass = 0.15;
cfg.nPass = 2;

cfg.dedupeDXyMax = 10.0;
cfg.dedupeRMin = 0.6;

cfg.xyCoarseHalf = 9;
cfg.xyFineHalf = 2;
cfg.xyLocal = 8;
cfg.avgBin = 10;
cfg.zMin = -40.0;
cfg.zMax = 40.0;
cfg.zRefineHalf = 20.0;
cfg.refineRPass = 0.12;
cfg.parallaxRmsMax = 6.0;
end

function add_matlab_lib(matlabDir)
%ADD_MATLAB_LIB Put helper folders on the path (idempotent).
libDir = fullfile(matlabDir, 'lib');
subs = {'io', 'psf', 'localize', 'match', 'recover'};
for i = 1:numel(subs)
    p = fullfile(libDir, subs{i});
    if exist(p, 'dir')
        addpath(p);
    end
end
end
