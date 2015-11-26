function word = real_word( index,dict,story_id,dict_of_unknown )
%REAL_WORD Summary of this function goes here
%   Detailed explanation goes here
      if index==1
          word='';
      elseif  (index>=dict('Unknown1'))%&(index<dict('Unknown10')+1)
          word=dict_of_unknown{index-dict('Unknown1')+1,story_id};
      else
          for iiii=dict.keys()
           try
                if isequal(dict(cell2mat(iiii)),index)
                    word=cell2mat(iiii);
                    break
               
                end
               
           catch 
              continue 
           end
          end
      end

end

