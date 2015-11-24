% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

rng('shuffle')
addpath nn;
addpath memory;
base_dir = 'D:/360pan/Data/FieldDialog/copus/Taxi/'; % path to data
workSpaceSavePath = [base_dir,'Model.mat']
TrainFlag = 1;
test_err = zeros(20,1);
test_num = zeros(20,1);

if TrainFlag == 1
    sprintf('start QA Train!')
    % parse data
    f = dir(fullfile(base_dir,['Taxi_Train.txt']));
    data_path = {fullfile(base_dir,f(1).name)};

    f = dir(fullfile(base_dir,['Taxi_Test.txt']));
    test_data_path = {fullfile(base_dir,f(1).name)};

    dict = containers.Map;
    dict('nil') = 1;
    rdict = containers.Map;
    rdict('1') = 'nil';
    unknown_dict = containers.Map;
    unknown_rdict = containers.Map;
    [story, questions,qstory] = parseBabiTask(data_path, dict,rdict, false);
    for i =1:500
       unknownID = sprintf('unknown%d', i);
       dict(unknownID) = length(dict) + 1;
       rdict(int2str(dict(unknownID))) = unknownID;
    end
   
    % train and test
    config_babi;
    build_model;
    if linear_start
        train_linear_start;
    else
        train;
    end
    save(workSpaceSavePath)
end


load(workSpaceSavePath); 
 %add unknown words in dict

[test_story, test_questions, test_qstory] = parseTestTask(test_data_path, dict, rdict,unknown_dict, unknown_rdict,false);
sprintf('start QA Test!');
test;
test_err= test_error;
test_num=total_test_num;

sprintf('End %dth QA!')

test_err
test_num