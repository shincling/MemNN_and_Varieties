function word = real_word( index,dict )
%REAL_WORD Summary of this function goes here
%   Detailed explanation goes here
      if index==1
          word='';
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

