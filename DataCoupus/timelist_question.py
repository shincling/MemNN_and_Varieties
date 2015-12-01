# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

timelist_question=[]
timelist_question.append('请问您需要购买什么时间的机票？')
timelist_question.append('您需要预定哪个时间的机票？')
timelist_question.append('请告诉我您预计的出行时间是？')
timelist_question.append('请问您的出行时间是？')
timelist_question.append('请告知您的出行时间。')
timelist_question.append('您什么时候出发？')
timelist_question.append('您要订哪天的机票？')
timelist_question.append('您要订什么时候的机票？')
timelist_question.append('您想买什么时候？')
timelist_question.append('您什么时候走？')
timelist_question.append('何时出发？')
timelist_question.append('什么时候的？')
timelist_question.append('什么时间的？')
timelist_question.append('好的，请提供出行时间。')
timelist_question.append('时间是？')
timelist_question.append('几号的？')
timelist_question.append('您需要什么时候走？')
timelist_question.append('说一下您的出行时间，谢谢。')

timelist_question_cut=[]
for ans in timelist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    timelist_question_cut.append(w_sent)
pass