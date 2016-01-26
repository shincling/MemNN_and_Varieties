dict_order1={};
%to get the dict mapprintpython

    for index=1:length(dict)
        j=index
        for iiii=dict.keys()
            if isequal(dict(cell2mat(iiii)),index)
                dict_order1{index}=iiii;
                break
            end
        end
    end
out_list={};
% for ii = 1:1000
%     for i =1:8%16
%         out_list{ii,i}=dict_order1{outtt(2*i+16*(ii-1))};%(i+16*(ii-1))
%         i+16*(ii-1)
%     end
% end

for ii = 1:999
    for i =1:23
        
        if (length(dict)-9)<=outtt(i+23*(ii-1))&outtt(i+22*(ii-1))<=length(dict)
            out_list{ii,i}=dict_un{outtt(i+23*(ii-1))+10-length(dict),ii};
        else
            out_list{ii,i}=dict_order1{outtt(i+23*(ii-1))};
        end
        i+23*(ii-1)
    end
end

 