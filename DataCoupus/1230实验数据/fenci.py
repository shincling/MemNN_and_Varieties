# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa29_0104/qa29')
content=f.read()
content=content.replace('\r\n','\t')
content=content.split('\t')
w_sent=''
for i in range(len(content)):
    if i%2==1:

        sent=jieba._lcut(content[i])
        for word in (sent):
            w_sent +=' '
            w_sent +=word
        w_sent += '\n'

fw=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa29_0104/qa29_fenci','w')
fw.write(w_sent.encode('utf8'))
