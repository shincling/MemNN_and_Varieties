# -*- coding: utf8 -*-
__author__ = 'shin'
import re
import jieba
import random
from namelist_question import namelist_question_cut
from namelist_answer import namelist_answer_cut
from countlist_question import countlist_question_cut
from countlist_answer import countlist_answer_cut
from departurelist_question import departurelist_question_cut
from departurelist_answer import departurelist_answer_cut


print 'name question:%d'%len(namelist_question_cut)
print 'name answer:%d\n'%len(namelist_answer_cut)

storyNumber=1000
fw=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/ticket_shin.txt','w')

familyName=['张','王','李','赵','周','吴','顾','郑','何','万','黄','周','吴','徐','孙','胡','朱','高',
            '林','何','郭','马','罗','梁','宋','谢','韩','唐','冯','于','董','萧','程','曹','袁','邓',
            '许','欧阳','太史','端木','上官','司马','东方','独孤','南宫','万俟','闻人','夏侯','诸葛','尉迟','公羊']

lastName=['舒敏','安邦','安福', '安歌', '安国','刚捷', '刚毅', '高昂', '高岑', '高畅', '高超', '高驰', '高达','浩言' ,'皓轩', '和蔼' ,'和安' ,'和璧', '和昶', '和畅',
          '天' ,'景','同' ,'景曜' ,'靖','琪' ,'君昊', '君浩', '俊艾' ,'俊拔', '俊弼',  '才', '风' ,'歌' ,'光' ,'和', '平',  '和洽' ,'和惬' ,'和顺', '和硕', '和颂',
          '佳','可嘉','可','心','琨瑶','琨瑜','兰','芳','兰蕙','梦','娜','若','英','月','兰泽','芝','岚翠','风','岚岚','蓝','尹']


countDict=['一张','二张','两张','三张','四张','五张','六张','七张','八张','九张','十张']



def namePart(f,ind):

    fullname=random.choice(familyName)+random.choice(lastName)

    f.write('%d%s'%(ind+1,random.choice(namelist_question_cut).encode('utf8')))
    ans_sent=random.choice(namelist_answer_cut).replace('[slot_name]',fullname.decode('utf8'))
    f.write('%d%s'%(ind+2,ans_sent.encode('utf8')))

    f.write('%d count ?\tnil\t%d\n'%(ind+3,ind+2))
    f.write('%d name ?\t%s\t%d\n'%(ind+4,fullname,ind+2))
    f.write('%d destination ?\tnil\t%d\n'%(ind+5,ind+2))
    f.write('%d departure ?\tnil\t%d\n'%(ind+6,ind+2))
    f.write('%d idnumber ?\tnil\t%d\n'%(ind+7,ind+2))
    f.write('%d time ?\tnil\t%d\n'%(ind+8,ind+2))
    f.write('%d phone ?\tnil\t%d\n'%(ind+9,ind+2))

    ind=ind+10
    return f,ind

def countPart(f,ind):

    f.write('%d%s'%(ind+1,random.choice(countlist_question_cut).encode('utf8')))

    rand_or_rule=random.randint(0,1)#0的时候规则，1的时候随机
    if rand_or_rule:
        fullcount=str(random.randint(0,66666))+'张'
    else :
        fullcount=random.choice(countDict)
    ans_sent=random.choice(countlist_answer_cut).replace('[slot_count]',fullcount.decode('utf8'))
    f.write('%d%s'%(ind+2,ans_sent.encode('utf8')))

    f.write('%d count ?\t%s\t%d\n'%(ind+3,fullcount,ind+2))
    f.write('%d name ?\tnil\t%d\n'%(ind+4,ind+2))
    f.write('%d destination ?\tnil\t%d\n'%(ind+5,ind+2))
    f.write('%d departure ?\tnil\t%d\n'%(ind+6,ind+2))
    f.write('%d idnumber ?\tnil\t%d\n'%(ind+7,ind+2))
    f.write('%d time ?\tnil\t%d\n'%(ind+8,ind+2))
    f.write('%d phone ?\tnil\t%d\n'%(ind+9,ind+2))

    ind=ind+10
    return f,ind


def departurePart(f,ind):
    
    
    
    
    
    
    
    return

def destinationPart(ind):
    return

def timePart(f,ind):
    return

def idnumberPart(f,ind):
    return

def phonePart(f,ind):
    return


for story_ind in range(storyNumber):
    line_ind=1
    '''---------------greeting--------------'''

    fw.write('%d 您好 ， 机票 预订 中心 ， 需要 我 为 你 做些 什么 ？\n'%line_ind)
    line_ind+=1
    fw.write('%d 我 想 预订 机票 。\n'%line_ind)
    line_ind+=1

    '''---------------greeting--------------'''

    fw,line_ind=namePart(fw,line_ind)
    fw,line_ind=countPart(fw,line_ind)
    fw,line_ind=departurePart(fw,line_ind)
    
    
    fw,line_ind=destinationPart(fw,line_ind)
    fw,line_ind=timePart(fw,line_ind)
    fw,line_ind=idnumberPart(fw,line_ind)
    fw,line_ind=phonePart(fw,line_ind)




