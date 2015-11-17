% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

rng('shuffle')
addpath nn;
addpath memory;
base_dir = '/home/shin/DeepLearning/数据集/Facebook QA/tasks_1-20_v1-2/en'; % path to data
t = 24; % task ID
num_of_unknown=10;


% parse data
f = dir(fullfile(base_dir,['qa',num2str(t),'_*_train.txt']));
data_path = {fullfile(base_dir,f(1).name)};
f = dir(fullfile(base_dir,['qa',num2str(t),'_*_test.txt']));
test_data_path = {fullfile(base_dir,f(1).name)};
dict = containers.Map;
dict_unknown=cell(num_of_unknown,2000);
dict_unknown(:)={'00000000000'};


dict('nil') = 1;
[story, questions,qstory,dict_un] = parseBabiTask(data_path, dict, false,dict_unknown,num_of_unknown,1); %story:6*10*200 ,question:10*1000,qstory=6*1000

for u_i =[1:num_of_unknown]
    dict(strcat('Unknown',int2str(u_i))) = dict.length+1;
end
    
    
   
% dict('Unknown1') = dict.length+1;
% dict('Unknown2') = dict.length+1;
% dict('Unknown3') = dict.length+1;
% dict('Unknown4') = dict.length+1;
% dict('Unknown5') = dict.length+1;
% dict('Unknown6') = dict.length+1;
% dict('Unknown7') = dict.length+1;
% dict('Unknown8') = dict.length+1;
% dict('Unknown9') = dict.length+1;
% dict('Unknown10') = dict.length+1;


[test_story, test_questions, test_qstory,dict_un] = parseBabiTask(test_data_path, dict, false,dict_unknown,num_of_unknown,0); %test is the same

global wrong_index;
%wrong_index=zeros(2,32);
wrong_index=[];
% train and test
config_babi;
build_model;
if linear_start
    train_linear_start;
else
    train;
end
test;