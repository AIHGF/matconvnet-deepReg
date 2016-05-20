function runLSP2DPose(varargin)

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;
 
addpath('../networks');
addpath('../examples');

opts.datas='LSP';

opts.patchHei=120;
opts.patchWi=80;

opts.cam=1;%camera
opts.aug=30;%amount of augmentation

opts.batchSize = 230;
opts.numSubBatches = 1;
opts.numEpochs = 150 ;
opts.learningRate = 0.01;
opts.useBnorm = false ;
opts.prefetch = false ;

%GPU (leave it empty for training on CPU)
opts.gpus = [2];

opts.initNet=''; %pre-trained network
opts.outNode=28; %predicted-values
opts.inNode=3; %input channels
opts.lossFunc='tukeyloss'; %options: tukeyloss OR l2loss
opts.thrs=[];%not used
opts.refine=false;

%axis error plot (x,y)
opts.scbox=opts.patchWi*ones(opts.outNode,1);
opts.scbox(2:2:end)=opts.patchHei;

opts.expDir = sprintf('../data/train/%s-baseline_%s_%d',opts.datas,opts.lossFunc,opts.cam) ;
opts.imdbPath = sprintf('../data/train/%s-baseline_imdb%d.mat',opts.datas, opts.cam);%RAM image path

opts.DataMatTrain=sprintf('../data/%s/%s_imdbsT%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('../data/%s/%s_imdbsV%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);

%load network
opts.net = initializeRegNetwork(opts);

%objectives
opts.derOutputs = {'objective',1} ;

opts = vl_argparse(opts, varargin);

%train
cnn_regressor_dag(opts);
