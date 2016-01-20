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
for ii = 1:1000
    for i =1:8%16
        out_list{ii,i}=dict_order1{outtt(2*i+16*(ii-1))};%(i+16*(ii-1))
        i+16*(ii-1)
    end
end
    