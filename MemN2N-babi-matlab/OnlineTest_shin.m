function  response = OnlineTest_shin( story_path,slot_number)%,slot_number,slot_departure,slot_destination,slot_name,slot_idnumber,slot_time,slot_count) )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
addpath nn;
addpath memory;
dict = containers.Map;


ResultFilePath = '/home/shin/DeepLearning/MemoryNetwork/QA/Interface/OnlineTest/Result.txt';
base_dir = '/home/shin/DeepLearning/MemoryNetwork/QA/copus/'; % path to data
%base_dir = [base_dir, topic];
%workSpaceSavePath = [base_dir,'\Model.mat'];
%load(workSpaceSavePath); 

load confidence
%batch_size=slot_number
base_dir = '/home/shin/DeepLearning/MemoryNetwork/QA/Interface/OnlineTest';
f = dir(fullfile(base_dir,['Story.txt']));
fp = fopen(ResultFilePath, 'wt','n', 'utf-8');
StoryFilePath = {fullfile(base_dir,f(1).name)};
    
    include_question=false
    [online_story, online_questions, online_qstory,online_dict_un] = parseTestTask_shin(StoryFilePath, dict, include_question,dict_unknown,num_of_unknown);
    
    oushulie=online_story(:,2:2:end);
    oushulie=sort(oushulie(:));
    oushulie(end+1,1)=1;
    soft_area=unique(sort(oushulie(:)));
    
    
    
    
    online_story(end+1:max_words,:)=1;
    online_story(:,end+1:32)=1;
    online_qstory(end+1:max_words,:)=1;
    online_qstory(:,end+1:32)=1;
    online_questions(:,end+1:32)=1;
    
    batch = (1:batch_size) ;
    
    input = zeros(size(story,1),slot_number,'single');
   % input=[16,18,19,20,21,22;17,17,17,17,17,17;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1;1,1,1,1,1,1];
    %target = test_questions(3,batch);
    
    input(:) = dict('nil');
    memory{1}.data(:) = dict('nil');
    %memory{1}.data(:) = 1;
   
    % input=online_qstory
   
     % b=1;
    for b = 1:32%1:batch_size 
        d = online_story(:,1:online_questions(2,batch(b)),online_questions(1,batch(b)));
        d = d(:,max(1,end-config.sz+1):end);
        memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
        ddd(1,b)={d};
        if enable_time
            memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict); % time words
        end
        input(1:size(online_qstory,1),b) = online_qstory(:,batch(b));
    end
    for i = 2:nhops
        memory{i}.data = memory{1}.data;
    end
    
    out = model.fprop(input);
    out = out(soft_area,:);
    [ppp,m_index]=max(out(:,1:6));
    m_index=soft_area(m_index);  %shin 
   % cost = loss.fprop(out, target);
   out_word=cell(1,6);
   
   for qqqq=1:6
    for iiii=dict.keys()
           try
                if isequal(dict(cell2mat(iiii)),m_index(qqqq))
                    out_word(qqqq)=iiii;%cell2mat(iiii);
                    break
               
                end
               
           catch 
              continue 
           end
    end
   end
   
 for qqqq=1:6
    if (m_index(qqqq)>=dict('Unknown1'))&(m_index(qqqq)<(dict('Unknown10')+1))
        out_word(qqqq)=online_dict_un(m_index(qqqq)-dict('Unknown1')+1,1);
    end
    
    
    
    
    response = out_word
end

