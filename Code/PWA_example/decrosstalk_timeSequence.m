clear
clc
close all

rootDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(rootDir, 'data');
pathroot = fullfile(dataDir, 'PSF', 'PSF_stack');
filenamePre='PSF1_00004';
filename=[filenamePre,'.tif'];
disp('Loading file...')
path=fullfile(pathroot,filename);
P=loadtiff(path);
sizeP=size(P);
load(fullfile(dataDir, 'crosstalk', 'mtxS_Beads_1127.mat'))
% load('F:\TH\reflectSMMAS\smmas-decrosstalk-old\S.mat')
disp('Adjusting data format...')
S=S/max(max(S));
% S=S';
P4=reshape(P,[size(P,1),size(P,2),4,size(P,3)/4]);
P4f=permute(P4,[3,1,2,4]);
P4f=reshape(P4f,4,[]);
clearvars -EXCEPT S P4f sizeP pathroot filenamePre % Clear variables to avoid running out of memory
disp('Decrosstalk...')
N=1;% If memory is insufficient, split the raw data into N sequential batches
blockSize=floor(size(P4f,2)/N);
numPoints=size(P4f,2);
If=int16([0;0;0;0]);
for n=N:-1:1
    disp(['Processing batch ',num2str(n)])
    block=P4f(:,(n-1)*blockSize+1:min(n*blockSize,numPoints));
    If=[If,int16(flip(S\single(block),2))];
    P4f(:,(n-1)*blockSize+1:min(n*blockSize,numPoints))=[];
end
If(:,1)=[];
If=flip(If,2);
% If=S\single(P4f);
disp('Adjusting data format...')
I=reshape(If,[4,sizeP(1),sizeP(2),sizeP(3)/4]);
I=permute(I,[2,3,4,1]);

if ~exist(fullfile(pathroot,['dc_',filenamePre]),'dir')
    mkdir(fullfile(pathroot,['dc_',filenamePre]));
end
disp('Writing ...')
for i=1:4
    disp(['FOV',num2str(i),'...'])
    defilepath=fullfile(pathroot,['dc_',filenamePre],[filenamePre,'_',num2str(i),'.tif']);
    if exist(defilepath,'file')
        delete(defilepath)
    end
    saveastiff(I(:,:,:,i), defilepath)
end

disp('end')
