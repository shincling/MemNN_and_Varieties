# -*- coding: utf8 -*-
__author__ = 'shin'
import re
import jieba
from namelist_question import namelist_question_cut
from namelist_answer import namelist_answer_cut

print 'name question:%d'%len(namelist_question_cut)
print 'name answer:%d\n'%len(namelist_answer_cut)

storyNumber=1000
fw=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/ticket_shin.txt','w')

def namePart(f):

    familyName=['张','王','李','赵','周','吴','顾','郑','何','万','黄','周','吴','徐','孙','胡','朱','高',
                '林','何','郭','马','罗','梁','宋','谢','韩','唐','冯','于','董','萧','程','曹','袁','邓',
                '许','欧阳','太史','端木','上官','司马','东方','独孤','南宫','万俟','闻人','夏侯','诸葛','尉迟','公羊']

    realName=['舒敏','安邦','安福', '安歌', '安国','刚捷', '刚毅', '高昂', '高岑', '高畅', '高超', '高驰', '高达','浩言' ,'皓轩', '和蔼' ,'和安' ,'和璧', '和昶', '和畅',
              '天' ,'景','同' ,'景曜' ,'靖','琪' ,'君昊', '君浩', '俊艾' ,'俊拔', '俊弼',  '才', '风' ,'歌' ,'光' ,'和', '平',  '和洽' ,'和惬' ,'和顺', '和硕', '和颂',
              '佳','可嘉','可','心','琨瑶','琨瑜','兰','芳','兰蕙','梦','娜','若','英','月','兰泽','芝','岚翠','风','岚岚','蓝','尹']
    return



for story_ind in range(storyNumber):

    '''---------------greeting--------------'''

    fw.write('1 您好 ， 机票 预订 中心 ， 需要 我 为 你 做些 什么 ？\n')
    fw.write('2 我 想 预订 机票 。\n')

    '''---------------greeting--------------'''

    namePart(fw)




