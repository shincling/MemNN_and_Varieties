# -*- coding: utf8 -*-
__author__ = 'shin'
import jieba

namelist_question=[]
namelist_question.append('您好，请问您的姓名是？')
namelist_question.append('请问您的姓名是？')
namelist_question.append('请告诉我您的姓名')
namelist_question.append('请您告诉我您的名字。')
namelist_question.append('请问您要购买机票的用户姓名是？')
namelist_question.append('请问您的名字是？')
namelist_question.append('请告知您的姓名。')
namelist_question.append('我们需要知道您的姓名。')
namelist_question.append('您怎么称呼？')
namelist_question.append('您的全名是什么？')
namelist_question.append('请提供您的全名。')
namelist_question.append('请输入您的全名。')
namelist_question.append('您叫什么名字啊？')
namelist_question.append('您的名字是什么？')
namelist_question.append('请告知您的名字。')
namelist_question.append('请问尊姓大名？')
namelist_question.append('请输入乘客姓名。')
namelist_question.append('乘客的名字是什么？')
namelist_question.append('乘客怎么称呼？')
namelist_question.append('乘客叫什么名字？')
namelist_question.append('乘客的姓名是？')
namelist_question.append('请问先生怎么称呼？')
namelist_question.append('请问小姐怎么称呼？')
namelist_question.append('请问老人家怎么称呼？')
namelist_question.append('先生您怎么称呼？')
namelist_question.append('小姐您怎么称呼？')
namelist_question.append('先生您叫什么名字？')
namelist_question.append('小姐您叫什么名字？')
namelist_question.append('您的名字？')
namelist_question.append('先生的名字？')
namelist_question.append('小姐的名字？')
namelist_question.append('乘客姓名？')
namelist_question.append('姓名？')
namelist_question.append('名字？')
namelist_question.append('可否请教先生名姓？')
namelist_question.append('小姐芳名可否见告？')
namelist_question.append('麻烦您说一下您的姓名可以吗？')
namelist_question.append('麻烦说下您的名字？谢谢。')
namelist_question.append('请告知姓名，谢谢。')
namelist_question.append('麻烦您告诉我您的名字，非常感谢。')

namelist_question_cut=[]
for ans in namelist_question:
    w_sent=''
    sent=jieba._lcut(ans)
    for word in (sent):
        w_sent +=' '
        w_sent +=word
    w_sent += '\n'
    namelist_question_cut.append(w_sent)
pass