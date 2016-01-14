% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.


global wrong_index;
global out_presentation;
total_test_err = 0;
total_test_num = 0;
ddd=cell(1,batch_size);


outtt=[];
out_presentation=0;
for k = 1:floor(size(test_questions,2)/batch_size)
    batch = (1:batch_size) + (k-1) * batch_size;
    input = zeros(size(story,1),batch_size,'single');
    target = test_questions(3,batch);
    input(:) = dict('nil');
    memory{1}.data(:) = dict('nil');
    for b = 1:batch_size
        d = test_story(:,1:test_questions(2,batch(b)),test_questions(1,batch(b)));
        d = d(:,max(1,end-config.sz+1):end);
        %---------------------shin-----------------------�������slot״̬ȥ��
%         dellist=[];
%         if size(d,2)>3
%             for j =2:3:(size(d,2)-4)
%                 dellist=[dellist j];
%             end
%             d(:,dellist)=[];
%         end
        %--------------------shin----------------------
        
         
        
        
        
        
        
        memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
        ddd(1,b)={d};
        if enable_time
            memory{1}.data(end,1:size(d,2),b) = (size(d,2):-1:1) + length(dict); % time words
        end
        input(1:size(test_qstory,1),b) = test_qstory(:,batch(b));
    end
    for i = 2:nhops
        memory{i}.data = memory{1}.data;
    end
    
    out = model.fprop(input);
    [~,yyy] = max(out,[],1);
    outtt=[outtt,yyy];
    cost = loss.fprop(out, target);
    total_test_err = total_test_err + loss.get_error(out, target);
    
% 
%      for aaa =[1:size(wrong_index,2)]
%          fprintf('The story is : \n');
%          for sss_index =wrong_index(3,aaa)
%              evi=cell2mat(ddd(1,sss_index));
%                 for sent_index=1:size(evi,2)
%                     sent=evi(:,sent_index);
%                     for word=1:size(sent)
%                         fprintf(real_word(sent(word),dict,test_questions(1,batch(sss_index)),dict_un));
%                         fprintf('    ');
%                     end
%                     fprintf('\n');
%                 end
%          
%          end
%          
%          fprintf('The question is : \n');
%          for sss_index =wrong_index(3,aaa)
%              for q_sent=input(:,sss_index);
%                  for q_word=1:size(q_sent)
%                      fprintf(real_word(q_sent(q_word),dict));
%                      fprintf('    ');
%                  end
%                  fprintf('\n');
%              end
%          end
% 
%          for iii=dict.keys()
%                         try
%                                 if isequal(dict(cell2mat(iii)),wrong_index(1,aaa))
%                                    
%                                     generation=iii;
%                                 end
%                                 if isequal(dict(cell2mat(iii)),wrong_index(2,aaa))
%                                     
%                                     target=iii;
%                                 end
%                         catch 
%                             continue 
%                         end
%          end
%          
%          
%          fprintf('\nThe target is : \n %s',cell2mat(target));
%          fprintf('\nThe answer is : \n %s  ',cell2mat(generation));     
%          
%          
%          
%          
%          fprintf('\n---------------------------------------------------------------------\n');
%      end
    
    
    
    
    
    total_test_num = total_test_num + batch_size;
end

test_error = total_test_err/total_test_num;
%target_list;

disp(['test error: ', num2str(test_error)]);
