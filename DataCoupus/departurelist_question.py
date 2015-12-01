# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

departurelist_question=[]
departurelist_question.append('请问您需要购买什么时间的机票？')
departurelist_question.append('您需要预定哪个时间的机票？')
departurelist_question.append('请告诉我您预计的出行时间是？')
departurelist_question.append('请问您的出行时间是？')
departurelist_question.append('请告知您的出行时间。')
departurelist_question.append('您从哪里走？')
departurelist_question.append('您从哪里出发呢？')
departurelist_question.append('您从哪里起飞？')
departurelist_question.append('说下您的出发地？')
departurelist_question.append('您旅行的起点是哪里？')
departurelist_question.append('您从哪座城市出发？')
departurelist_question.append('从哪走？')
departurelist_question.append('在哪里起飞？')
departurelist_question.append('您要买从哪里出发的机票？')
#departurelist_question.append('先生从哪里出发？')
#departurelist_question.append('小姐从哪里出发？')
departurelist_question.append('好的，麻烦说一下起点是哪里？')

departurelist_question_cut=[]
for ans in departurelist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    departurelist_question_cut.append(w_sent)
pass