# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

departurelist_answer=[]
'''
departure=<departure_Answer>
departure=<departure_Answer>就够了。
departure=我要<departure_Answer>机票。
departure=我想订<departure_Answer>。
departure=买<departure_Answer>票就可以。
departure=机票数量是<departure_Answer>。
departure=想订购<departure_Answer>机票。
departure=<departure_Answer>即可。
departure=可能需要<departure_Answer>。

[槽_departure]。
[槽_departure]张就可以了。
帮我预定[槽_departure]张。
[槽_departure]张。
[槽_departure]张机票。
'''

departurelist_answer.append('北京')#这里的trick是加大一下常用句式的比重
departurelist_answer.append('北京')
departurelist_answer.append('北京')
departurelist_answer.append('北京')

departurelist_answer.append('从北京起飞。')
departurelist_answer.append('从北京出发的机票')
departurelist_answer.append('机票的出发地是北京。')
departurelist_answer.append('从北京出发。')
departurelist_answer.append('自北京出发。')
departurelist_answer.append('由北京起飞的飞机。')
departurelist_answer.append('北京。')
departurelist_answer.append('时间是北京。')
departurelist_answer.append('帮我预订北京的机票。')
departurelist_answer.append('出行地点是北京。')
departurelist_answer.append('订从北京走的机票。')
departurelist_answer_cut=[]

for ans in departurelist_answer:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    w_sent=w_sent.replace('北京'.decode('utf8'),'[slot_departure]')
    departurelist_answer_cut.append(w_sent)
pass