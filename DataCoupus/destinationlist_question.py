# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

destinationlist_question=[]
destinationlist_question.append('请问您的目的地是哪里？')
destinationlist_question.append('请问您要预定的目的地是？')
destinationlist_question.append('请问您要预定飞往哪个城市的机票？')
destinationlist_question.append('请问您此次出行的目的地是？')
destinationlist_question.append('请问您此次前往哪个城市？')
destinationlist_question.append('您要去什么地方？')
destinationlist_question.append('您要飞往什么地方？')
destinationlist_question.append('您要飞往哪座城市？')
destinationlist_question.append('您想订去哪里的飞机票？')
destinationlist_question.append('去哪？')
destinationlist_question.append('去哪儿？')
destinationlist_question.append('往哪里去？')
destinationlist_question.append('去到哪里？')
destinationlist_question.append('您机票的目的地是哪里？')
destinationlist_question.append('目的地？')
destinationlist_question.append('终点是？')
destinationlist_question.append('您的目的地？')
destinationlist_question.append('您要订飞往那里的票？')
destinationlist_question.append('去哪里的票？')
#destinationlist_question.append('先生订去哪的票呀？')
#destinationlist_question.append('小姐订去哪的票呀？')
#destinationlist_question.append('先生去哪里？')
#destinationlist_question.append('小姐去哪里？')
destinationlist_question.append('请告诉我您旅行的目的地。')
destinationlist_question.append('您告诉我您的目的地。')
destinationlist_question.append('没问题，麻烦说下目的地。')

destinationlist_question_cut=[]
for ans in destinationlist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    destinationlist_question_cut.append(w_sent)
pass