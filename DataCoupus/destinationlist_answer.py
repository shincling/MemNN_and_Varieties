# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

destinationlist_answer=[]
'''
destinationlist_answer.append('去[slot_destination]。')
destinationlist_answer.append('目的地是[slot_destination]。')
destinationlist_answer.append('飞往[slot_destination]。')
destinationlist_answer.append('是[slot_destination]。')
destinationlist_answer.append('[slot_destination]')
destinationlist_answer.append('去[slot_destination]。')
destinationlist_answer.append('往[slot_destination]。')
destinationlist_answer.append('到[slot_destination]去。')
destinationlist_answer.append('[slot_destination]是我的目的地。')
destinationlist_answer.append('目的地是[slot_destination]。')
destinationlist_answer.append('我要到[slot_destination]去。')
destinationlist_answer.append('飞往[slot_destination]的飞机。')
destinationlist_answer.append('买去[slot_destination]的机票。')
destinationlist_answer.append('我要订去[slot_destination]的飞机。')
'''

destinationlist_answer.append('北京')#这里的trick是加大一下常用句式的比重
destinationlist_answer.append('北京')
destinationlist_answer.append('北京')
destinationlist_answer.append('北京')
destinationlist_answer.append('北京。')
destinationlist_answer.append('北京。')
destinationlist_answer.append('北京。')
destinationlist_answer.append('北京。')


destinationlist_answer.append('去北京。')
destinationlist_answer.append('目的地是北京。')
destinationlist_answer.append('飞往北京。')
destinationlist_answer.append('是北京。')
destinationlist_answer.append('北京')
destinationlist_answer.append('去北京。')
destinationlist_answer.append('往北京。')
destinationlist_answer.append('到北京去。')
destinationlist_answer.append('北京是我的目的地。')
destinationlist_answer.append('目的地是北京。')
destinationlist_answer.append('我要到北京去。')
destinationlist_answer.append('飞往北京的飞机。')
destinationlist_answer.append('买去北京的机票。')
destinationlist_answer.append('我要订去北京的飞机。')

destinationlist_answer.append('北京')
destinationlist_answer.append('我要去北京')
destinationlist_answer.append('到北京')
destinationlist_answer.append('我到北京')
destinationlist_answer.append('我去北京')
destinationlist_answer.append('到北京的机票')
destinationlist_answer.append('飞北京的机票')
destinationlist_answer.append('到北京的票')
destinationlist_answer.append('飞北京的票')
destinationlist_answer.append('去北京的机票')
destinationlist_answer.append('去北京的票')

destinationlist_answer_cut=[]

for ans in destinationlist_answer:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    w_sent=w_sent.replace('北京'.decode('utf8'),'[slot_destination]')
    destinationlist_answer_cut.append(w_sent)
pass