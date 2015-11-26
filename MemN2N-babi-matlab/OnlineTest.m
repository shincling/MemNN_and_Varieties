
function  OnlineTest(topic)

addpath nn;
addpath memory;


ResultFilePath = '/home/shin/DeepLearning/MemoryNetwork/QA/Interface/OnlineTest/Result.txt';
base_dir = '/home/shin/DeepLearning/MemoryNetwork/QA/copus/'; % path to data
base_dir = [base_dir, topic];
%workSpaceSavePath = [base_dir,'\Model.mat'];
%load(workSpaceSavePath); 
load just_1.mat;

base_dir = '/home/shin/DeepLearning/MemoryNetwork/QA/Interface/OnlineTest';
f = dir(fullfile(base_dir,['Story.txt']));
StoryFilePath = {fullfile(base_dir,f(1).name)};

fp = fopen(ResultFilePath, 'wt','n', 'utf-8');

% rdict={};
% unknown_dict={};
% unknown_rdict={};

[test_story, test_questions, test_qstory] = parseTestTask(StoryFilePath, dict, rdict,unknown_dict, unknown_rdict, false);

batch_size = size(test_questions, 2);
batch = 1:batch_size;
input = zeros(size(story,1),32,'single');
input(:) = dict('nil');
memory{1}.data(:) = dict('nil');
for b = 1:batch_size
    d = test_story(:,1:test_questions(2,batch(b)),test_questions(1,batch(b)));
    d = d(:,max(1,end-config.sz+1):end);
    memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
    if enable_time
        memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict); % time words
    end
    input(1:size(test_qstory,1),b) = test_qstory(:,batch(b));
end
for i = 2:nhops
    memory{i}.data = memory{1}.data;
end

out = model.fprop(input);

%[maxV, index] = max(out);


for i = 1:batch_size
    % get story
    d = test_story(:,1:test_questions(2,batch(i)),test_questions(1,batch(i)));
    d = d(:,max(1,end-config.sz+1):end);

    d1 = d(:,end);
    %d1(d1==1)=[];
    storyWord_index = unique(d1);
    storyWord_indicator = zeros(size(out, 1), 1);
    storyWord_indicator(storyWord_index) = 1;
    query_out = out(:, i) .* storyWord_indicator;
    [maxV, index] = max(query_out);


    % write result
    TrueValue = rdict(int2str(target(i)));
    if index >= length(rdict)
        outValue = 'nil';
    else   
        outValue = rdict(int2str(index));
        if length(strfind(outValue, 'unknown')) > 0
            outValue = unknown_rdict(outValue);
        end
    end

    
    
    query = input(:, i);
    query(find(query==0))=[];
    query(find(query==1))=[];
    for m = 1 : length(query)
        fprintf(fp, '%s', rdict(int2str(query(m))));
    end
    fprintf(fp, '=');  
    fprintf(fp, '%s\n', outValue);
    %fprintf(fp, '-----------------------------------------------\n')
end

return



