# -*- coding: utf8 -*-
__author__ = 'shin'

import matlab.engine

Matlabpath='/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/out_presentation'

eng = matlab.engine.start_matlab()
eng.eval('cd %s;' % (Matlabpath), nargout=0)

f_status=open('svm_train_status_1000','w')
f_next=open('svm_train_next_1000','w')
counter=0
for i in range(1,501):
    result=eng.eval("load('%d.mat')"%i)['output']

    for step_batch in range(32):
        '''
        for j in range(50):
            one_example=result[j][step_batch]
            if step_batch%2==0:
                f_status.write('%d:%f '%(j+1,result[j][(step_batch)/2]))
            if step_batch%2==1:
                f_next.write('%d:%f '%(j+1,result[j][(step_batch+1)/2]))
        '''
        if step_batch%2==0:
            for j in range(50):
                f_status.write('%d:%f '%(j+1,result[j][(step_batch)]))
                counter+=1
            f_status.write('\n')
        if step_batch%2==1:
            for j in range(50):
                f_next.write('%d:%f '%(j+1,result[j][step_batch]))
                counter+=1
            f_next.write('\n')


f_status.close()
f_next.close()
print counter


