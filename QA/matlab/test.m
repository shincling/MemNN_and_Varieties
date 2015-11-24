% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

DestPath= 'D:\Error.txt';
fp = fopen(DestPath, 'wt');


total_test_err = 0;
total_test_num = 0;
for k = 1:floor(size(test_questions,2)/batch_size)
%for k = 1:2
    batch = (1:batch_size) + (k-1) * batch_size;
    input = zeros(size(story,1),batch_size,'single');
    target = test_questions(3,batch);
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
    
    
    for i =1:batch_size
        % get story
        d = test_story(:,1:test_questions(2,batch(i)),test_questions(1,batch(i)));
        d = d(:,max(1,end-config.sz+1):end);
        
        storyWord_index = unique(d(:, end));
        storyWord_indicator = zeros(size(out, 1), 1);
        storyWord_indicator(storyWord_index) = 1;
        query_out = out(:, i) .* storyWord_indicator;
        [maxV, index] = max(query_out);
        
        
        if(index ~= target(i) && index ~= 1)
            TrueValue = rdict(int2str(target(i)));
            if index >= length(rdict)
                outValue = 'nil';
            else   
                outValue = rdict(int2str(index));
                if length(strfind(outValue, 'unknown')) > 0
                    outValue = unknown_rdict(outValue);
                end
                    
            end
            
            fprintf(fp, '-----------------------------------------------\n');
            fprintf(fp, '%dth Dialog, story is:\n', test_questions(1,batch(i)));
            
            d(find(d==0))=[];
            for m = 1 : size(d, 2)
                fprintf(fp, '%d\t', m);
                d_removeZeros = d(:, m);
                d_removeZeros(d_removeZeros==1)= [];
                for n = 1 : length(d_removeZeros)
                    value = rdict(int2str(d_removeZeros(n)));
                    if length(strfind(value, 'unknown')) > 0
                        value = unknown_rdict(value);
                    end
                    fprintf(fp, '%s ', value);
                end
                fprintf(fp, '\n');
            end
            fprintf(fp, 'query is: ');
            query = input(:, i);
            query(find(query==0))=[];
            query(find(query==1))=[];
            for m = 1 : length(query)
                fprintf(fp, '%s ', rdict(int2str(query(m))));
            end
            fprintf(fp, '?\n');
            fprintf(fp, 'ANSWER is: \n');  
            fprintf(fp, '**********RIGHT:%s\n**********WRONG:%s\n', TrueValue, outValue);
            %fprintf(fp, '-----------------------------------------------\n')
        end
        
    end
    cost = loss.fprop(out, target);
    total_test_err = total_test_err + loss.get_error(out, target);
    total_test_num = total_test_num + batch_size;
end



test_error = total_test_err/total_test_num;
disp(['test error: ', num2str(test_error)]);

