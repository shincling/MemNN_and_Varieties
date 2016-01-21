function[lstms,all_h_t,all_c_t]=Forward(batch,parameter,isTraining,h_from_MemNN)%Forward
    N=size(batch.Word,1);
    zeroState=zeroMatrix([parameter.hidden,N]); %1000*32的零矩阵
    if isTraining==1
        T=batch.MaxLen;%257这里
    else
        T=batch.MaxLenSource;
    end
    all_h_t=cell(parameter.layer_num,T);
    all_c_t=cell(parameter.layer_num,T);
    lstms = cell(parameter.layer_num,T);

    for ll=1:parameter.layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
            all_c_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
        end
    end
    for t=1:T %在每一个时刻（对于句子里的每一个词）
%         t=t
        for ll=1:parameter.layer_num %对于每一层，总共四层
%             if t<batch.MaxLenSource+1;%先对每一层进行初始化，target和source用不同的参数
%                 W=parameter.W_S{ll};
%             else
%                 W=parameter.W_T{ll};
%             end
            W=parameter.W_T{ll};
            if t==1
                    h_t_1=h_from_MemNN;
                    c_t_1 =zeroState;                    
            else
                c_t_1 = all_c_t{ll, t-1};
                h_t_1 = all_h_t{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.Word(:,t));%第一层查表给参数赋值embding
            else
                x_t=all_h_t{ll-1,t};%之后层是上一层的输出
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            [lstms{ll, t},all_h_t{ll, t},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);%LSTM unit calculation
        end
    end
end


