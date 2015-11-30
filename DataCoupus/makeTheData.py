# -*- coding: utf8 -*-
__author__ = 'shin'
import re
import jieba
from namelist_question import namelist_question
from namelist_answer import namelist_answer

print 'name question:%d'%len(namelist_question)
print 'name answer:%d\n'%len(namelist_answer)

storyNumber=1000
fw=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/ticket_shin.txt','w')

for story_ind in range(storyNumber):
    fw.write('1 您好 ， 机票 预订 中心 ， 需要 我 为 你 做些 什么 ？\n')
    fw.write('2 我 想 预订 机票 。\n')





