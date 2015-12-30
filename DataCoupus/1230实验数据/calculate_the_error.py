# -*- coding: utf8 -*-
__author__ = 'shin'
import re
import jieba
import random
import datetime

namelist=[395,237,283,432,370,137,388,453,447,270,407,378,190,350,308,205,422,20,280,297,261,231,306,213,457,161,459,364,420,75,383,117,112,428,325,179,454,443,390,424]
countlist=[224,417,434,209,134,223,393,95,286,396,128,103,259,415,273,197,367,359,289,406,157,194,148,419,401,249,55,235,380,439,321,337,351,448,243,412,430]
departurelist=[330,140,39,77,150,236,122,196,254,339,101,184,316,155,381]
destinationlist=[315,142,397,272,220,309,314,98,257,389,335,301,118,64,296,376,160,450,202,186,228,87]
timelist=[322,108,225,260,250,27,172,84,357,152,240,303,282,294,126,440,219,387]
idnumberlist=[48,426,230,374,368,200,176,115,368,452,252,207,320,166,82,284,363,342,354,334,372,348,290,274,132,458,298,463,278,266,442,145]
phonelist=[418,435,127,356,168,188,345,248,168,352,256,431,311,143,267,366,143,366,264,106,215,386,293,421,92,438,399,169,391,369,275,358,418,404,416,288,33]
totallist=[]
totallist.extend(namelist)
totallist.extend(countlist)
totallist.extend(departurelist)
totallist.extend(destinationlist)
totallist.extend(timelist)
totallist.extend(idnumberlist)
totallist.extend(phonelist)

f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/一起的数据.txt','r')
out=f.read()
out.split('\t')
ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa28_ticket_randOrder_ANS_slot_test.txt','r')
storys=ff.read().split('谢谢 。\n1 ')
one_story_list=[]
one_story_list.append(storys[0])
for i in range(1,len(storys)):
    one_story_list.append('1 '+storys[i])
assert one_story_list==1000

pass