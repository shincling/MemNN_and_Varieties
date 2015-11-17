% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

function [story, questions, qstory,dict_un] = parseBabiTask(data_path, dict, include_question,dict_unknown,num_of_unknown,train_or_test)
if train_or_test==1
        story = zeros(25, 1000, 1000, 'single');
        story_ind = 0;
        sentence_ind = 0;
        max_words = 0;
        max_sentences = 0;

        questions = zeros(10,1000, 'single');
        question_ind = 0;

        qstory = zeros(25,1000, 'single');

        fi = 1;
        fd = fopen(data_path{fi});
        line_ind = 0;

        while true
            line = fgets(fd);
            if ischar(line) == false
                fclose(fd);
                if fi < length(data_path)
                    fi = fi + 1;
                    fd = fopen(data_path{fi});
                    line_ind = 0;
                    line = fgets(fd);
                else
                    break
                end
            end
            line_ind = line_ind + 1;
            words = textscan(line, '%s');
            words = words{1}; %��line��仰 ת����Ϊ length*1��cell

            if strcmp(words{1}, '1')
                story_ind = story_ind + 1;%story index
                sentence_ind = 0;
                map = [];
            end

            if sum(line == '?') == 0  % not question
                is_question = false;
                sentence_ind = sentence_ind + 1;
            else        
                is_question = true;
                question_ind = question_ind + 1;
                questions(1,question_ind) = story_ind;
                questions(2,question_ind) = sentence_ind;
                if include_question
                    sentence_ind = sentence_ind + 1;
                end
            end

            map(end+1) = sentence_ind;

            for k = 2:length(words);
                w = words{k};
                w = lower(w);
                if w(end) == '.' || w(end) == '?'
                    w = w(1:end-1);
                end        
                if isKey(dict, w) == false
                    dict(w) = length(dict) + 1;
                end        
                max_words = max(max_words, k-1);%һ�仰����󳤶� 

                if is_question == false
                    story(k-1, sentence_ind, story_ind) = dict(w);
                else
                    qstory(k-1, question_ind) = dict(w);
                    if include_question == true
                        story(k-1, sentence_ind, story_ind) = dict(w);
                    end            

                    if words{k}(end) == '?'
                        answer = words{k+1};
                        answer = lower(answer);
                        if isKey(dict, answer) == false
                            dict(answer) = length(dict) + 1;
                        end
                        questions(3,question_ind) = dict(answer);
                        for h = k+2:length(words)
                            questions(2+h-k,question_ind) = map(str2num(words{h}));
                        end
                        questions(10,question_ind) = line_ind;
                        break
                    end
                end
            end
            max_sentences = max(max_sentences, sentence_ind);
        end
        story = story(1:max_words, 1:max_sentences, 1:story_ind);
        questions = questions(:,1:question_ind);
        qstory = qstory(1:max_words,1:question_ind);

        story(story == 0) = dict('nil');
        qstory(qstory == 0) = dict('nil');
end

if train_or_test==0
        story = zeros(25, 1000, 1000, 'single');
        story_ind = 0;
        sentence_ind = 0;
        max_words = 0;
        max_sentences = 0;

        questions = zeros(10,1000, 'single');
        question_ind = 0;

        qstory = zeros(25,1000, 'single');

        fi = 1;
        fd = fopen(data_path{fi});
        line_ind = 0;

        while true
            line = fgets(fd);
            if ischar(line) == false
                fclose(fd);
                if fi < length(data_path)
                    fi = fi + 1;
                    fd = fopen(data_path{fi});
                    line_ind = 0;
                    line = fgets(fd);
                else
                    break
                end
            end
            line_ind = line_ind + 1;
            words = textscan(line, '%s');
            words = words{1}; %��line��仰 ת����Ϊ length*1��cell

            if strcmp(words{1}, '1')
                story_ind = story_ind + 1;%story index
                un_index=1;% by shin   
                sentence_ind = 0;
                map = [];
            end

            if sum(line == '?') == 0  % not question
                is_question = false;
                sentence_ind = sentence_ind + 1;
            else        
                is_question = true;
                question_ind = question_ind + 1;
                questions(1,question_ind) = story_ind;
                questions(2,question_ind) = sentence_ind;
                if include_question
                    sentence_ind = sentence_ind + 1;
                end
            end

            map(end+1) = sentence_ind;

            for k = 2:length(words);
                w = words{k};
                w = lower(w);
                if w(end) == '.' || w(end) == '?'
                    w = w(1:end-1);
                end        
                if isKey(dict, w) == false
                    %dict(w) = length(dict) + 1;
                    if ismember({w},dict_unknown(:,story_ind))
                        for iii =[1:num_of_unknown]
                            if isequal(dict_unknown(iii,story_ind),{w})
                                w=strcat('Unknown',int2str(iii));
                            end
                        end
                    else
                        dict_unknown(un_index,story_ind)={w};  % by shin
                        w=strcat('Unknown',int2str(un_index));
                        un_index=un_index+1;
                    end
                end        
                max_words = max(max_words, k-1);%һ�仰����󳤶� 

                if is_question == false
                    story(k-1, sentence_ind, story_ind) = dict(w);  
                else
                    qstory(k-1, question_ind) = dict(w);
                    if include_question == true
                        story(k-1, sentence_ind, story_ind) = dict(w);
                    end            

                    if words{k}(end) == '?'
                        answer = words{k+1};
                        answer = lower(answer);
                       
                        if isKey(dict, answer) == false         % by shin
                            if ismember(answer,dict_unknown(:,story_ind))
                                for iii =[1:num_of_unknown]
                                	if  isequal(dict_unknown(iii,story_ind),{answer})
                                        answer=strcat('Unknown',int2str(iii));
                                    end
                                end
                            else
                            	dict_unknown(un_index,story_ind)={answer}; 
                            	answer=strcat('Unknown',int2str(un_index));
                             	un_index=un_index+1;
                            end
                        end
                        
                        
                        
                        questions(3,question_ind) = dict(answer);
                        for h = k+2:length(words)
                            questions(2+h-k,question_ind) = map(str2num(words{h}));
                        end
                        questions(10,question_ind) = line_ind;
                        break
                    end
                end
            end
            max_sentences = max(max_sentences, sentence_ind);
        end
        story = story(1:max_words, 1:max_sentences, 1:story_ind);
        questions = questions(:,1:question_ind);
        qstory = qstory(1:max_words,1:question_ind);
        

        story(story == 0) = dict('nil');
        qstory(qstory == 0) = dict('nil');
end
dict_un=dict_unknown;
end
