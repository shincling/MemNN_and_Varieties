import jieba
import re

f=open('/home/shin/DeepLearning/MemoryNetwork/QA/planeplane.txt','r')
content=f.read().decode('gbk').encode('utf8')
fw=open('/home/shin/DeepLearning/MemoryNetwork/QA/planeplane_shin','w')
content=content.split('dialogue ')
for i in range(1,5000):
    sent_id=1
    dia=content[i][(content[i].index('\n')+1):-4]
    dia=dia.split('\r\n')
    w_dia=''
    for j in range(len(dia)):
        '''
        if j%2==0:
            sent_w=
            assert 'M' in dia[j]
        '''
        sent=dia[j][3:-7]


        if 'greeting' in sent:

            w_sent=str(sent_id)
            sent_id=sent_id+1
            sent_list=jieba._lcut(sent[:sent.index('\t')])
            for word in (sent_list):
                w_sent +=' '
                w_sent +=word

            w_dia=w_dia+w_sent+'\n'
        elif 'request' in sent:
            pass
        elif 'inform' in sent:
            w_sent=str(sent_id)
            sent_id=sent_id+1
            sent_list=jieba._lcut(sent[:sent.index('\t')])
            for word in (sent_list):
                w_sent +=' '
                w_sent +=word

            w_dia=w_dia+w_sent+'\n'

        if j%2==1:
            w_dia=w_dia+str(sent_id)+' departure ?\n'
            sent_id=sent_id+1
            w_dia=w_dia+str(sent_id)+' destination ?\n'
            sent_id=sent_id+1
            w_dia=w_dia+str(sent_id)+' name ?\n'
            sent_id=sent_id+1
            w_dia=w_dia+str(sent_id)+' idnumber ?\n'
            sent_id=sent_id+1
            w_dia=w_dia+str(sent_id)+' time ?\n'
            sent_id=sent_id+1
            w_dia=w_dia+str(sent_id)+' count ?\n'
            sent_id=sent_id+1

















    pass
