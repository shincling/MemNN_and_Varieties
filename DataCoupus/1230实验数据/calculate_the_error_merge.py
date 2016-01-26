# -*- coding: utf8 -*-
__author__ = 'shin'
import re
import jieba
import random
import datetime
'''
namelist=[395,237,283,432,370,137,388,453,447,270,407,378,190,350,308,205,422,20,280,297,261,231,306,213,457,161,459,364,420,75,383,117,112,428,325,179,454,443,390,424]
countlist=[224,417,434,209,134,223,393,95,286,396,128,103,259,415,273,197,367,359,289,406,157,194,148,419,401,249,55,235,380,439,321,337,351,448,243,412,430]
departurelist=[330,140,39,77,150,236,122,196,254,339,101,184,316,155,381]
destinationlist=[315,142,397,272,220,309,314,98,257,389,335,301,118,64,296,376,160,450,202,186,228,87]
timelist=[322,108,225,260,250,27,172,84,357,152,240,303,282,294,126,440,219,387]
idnumberlist=[48,426,230,374,368,200,176,115,368,452,252,207,320,166,82,284,363,342,354,334,372,348,290,274,132,458,298,463,278,266,442,145]
phonelist=[418,435,127,356,168,188,345,248,168,352,256,431,311,143,267,366,143,366,264,106,215,386,293,421,92,438,399,169,391,369,275,358,418,404,416,288,33]
'''
countlist=[
        '您要订几张？',
        '您需要几张？',
        '您要多少张？',
        '您买几张？',
        '您要买多少张？',
        '您想买几张？',
        '您想要买多少张？',
        '您买几张票？',
        '先生买几张票？',
        '小姐买几张票？',
        '您需要订购多少张票？',
        '要买几张飞机票？',
        '几张？',
        '多少张？',
        '几张票？',
        '多少张票？',
        '需要票的数量是多少？',
        '您订购的数量？',
        '机票数量？',
        '您订购的数目是？',
        '飞机票数目？',
        '买几张？',
        '订几张？',
        '订购数量？',
        '您要订几张？谢谢。',
        '好的，请告诉我您要订几张？',
        'ok，没问题，订几张？',
        '您需要几张飞机票？',
        '请告诉我您需要多少张飞机票呢？',
        '您想要几张飞机票呢？',
        '您需要多少？谢谢。',
        '麻烦您说下买几张。',
        '知道了，买几张呀？',
        '明白了，麻烦说下买几张？',
        '好的，没问题，想要多少张呀？',
        '订机票吗，订几张呀？',
        '您要买多少张呢，谢谢。']

departurelist=[
        '请问您从哪里起飞？',
        '请问您从哪里出行？',
        '请问您从哪个城市出行？',
        '请告诉我您的起飞城市。',
        '请您告诉我，您从哪个城市出行？',
        '您从哪里走？',
        '您从哪里出发呢？',
        '您从哪里起飞？',
        '说下您的出发地？',
        '您旅行的起点是哪里？',
        '您从哪座城市出发？',
        '从哪走？',
        '在哪里起飞？',
        '您要买从哪里出发的机票？',
        '好的，麻烦说一下起点是哪里？']

destinationlist=[
        '请问您的目的地是哪里？',
        '请问您要预定的目的地是？',
        '请问您要预定飞往哪个城市的机票？',
        '请问您此次出行的目的地是？',
        '请问您此次前往哪个城市？',
        '您要去什么地方？',
        '您要飞往什么地方？',
        '您要飞往哪座城市？',
        '您想订去哪里的飞机票？',
        '去哪？',
        '去哪儿？',
        '往哪里去？',
        '去到哪里？',
        '您机票的目的地是哪里？',
        '目的地？',
        '终点是？',
        '您的目的地？',
        '您要订飞往那里的票？',
        '去哪里的票？',
        '请告诉我您旅行的目的地。',
        '您告诉我您的目的地。',
        '没问题，麻烦说下目的地。']

idnumberlist=[
        '请问您的身份证号码是多少？',
        '请问您的身份证号是？',
        '请告诉我您的身份证号码？',
        '您好，我需要输入您的身份证号码。',
        '请您告诉我您的身份证号码。',
        '请提供乘客的身份证号码。',
        '乘客的身份证是？',
        '您的身份证是多少啊？',
        '请您告诉我您的身份证号码。',
        '先生的身份证号码是多少呢？',
        '您身份证多少？',
        '您身份证号码多少？',
        '我们需要知道您的身份证号。',
        '身份证？',
        '身份证号？',
        '身份证号码？',
        '身份证件？',
        '身份证件号码？',
        '您身份证的号码？',
        '请告诉我您的身份证号码。',
        '小姐的身份证号码是多少呢？',
        '还需要提供身份证件，谢谢。',
        '身份证号？谢谢。',
        '麻烦您说下身份证号。',
        '请问您身份证，谢谢。',
        '好的，请告诉我您的身份证号码，非常感谢。',
        '没问题，您身份证件号码？',
        '知道了，您身份证的号码？',
        '明白了，请告诉我您的身份证号码。',
        '好的，小姐的身份证号码是多少呢？',
        '了解了，还需要提供身份证件，谢谢。',
        '嗯嗯，身份证号？谢谢。']

namelist=[
        '您好，请问您的姓名是？',
        '请问您的姓名是？',
        '请告诉我您的姓名',
        '请您告诉我您的名字。',
        '请问您要购买机票的用户姓名是？',
        '请问您的名字是？',
        '请告知您的姓名。',
        '我们需要知道您的姓名。',
        '您怎么称呼？',
        '您的全名是什么？',
        '请提供您的全名。',
        '请输入您的全名。',
        '您叫什么名字啊？',
        '您的名字是什么？',
        '请告知您的名字。',
        '请问尊姓大名？',
        '请输入乘客姓名。',
        '乘客的名字是什么？',
        '乘客怎么称呼？',
        '乘客叫什么名字？',
        '乘客的姓名是？',
        '请问先生怎么称呼？',
        '请问小姐怎么称呼？',
        '请问老人家怎么称呼？',
        '先生您怎么称呼？',
        '小姐您怎么称呼？',
        '先生您叫什么名字？',
        '小姐您叫什么名字？',
        '您的名字？',
        '先生的名字？',
        '小姐的名字？',
        '乘客姓名？',
        '姓名？',
        '名字？',
        '可否请教先生名姓？',
        '小姐芳名可否见告？',
        '麻烦您说一下您的姓名可以吗？',
        '麻烦说下您的名字？谢谢。',
        '请告知姓名，谢谢。',
        '麻烦您告诉我您的名字，非常感谢。']

phonelist=[
        '请问您的电话号码是多少？',
        '请问您的电话号是？',
        '请告诉我您的电话号码？',
        '您好，我需要输入您的电话号码。',
        '请您告诉我您的电话号码。',
        '请提供乘客的电话号码。',
        '乘客的电话是？',
        '您的电话是多少啊？',
        '请您告诉我您的电话号码。',
        '先生的电话号码是多少呢？',
        '您电话多少？',
        '您电话号码多少？',
        '我们需要知道您的电话号。',
        '电话？',
        '电话号？',
        '电话号码？',
        '电话？',
        '电话号码？',
        '您电话的号码？',
        '请告诉我您的电话号码。',
        '小姐的电话号码是多少呢？',
        '还需要提供电话，谢谢。',
        '电话号？谢谢。',
        '麻烦您说下电话号。',
        '请问您电话，谢谢。',
        '好的，请告诉我您的电话号码，非常感谢。',
        '没问题，您电话号码？',
        '知道了，您电话的号码？',
        '明白了，请告诉我您的电话号码。',
        '好的，小姐的电话号码是多少呢？',
        '了解了，还需要提供电话，谢谢。',
        '嗯嗯，电话号？谢谢。',
        '请问您的电话号码是多少？',
        '请告诉我您的联系方式？',
        '请问您的联系电话是？',
        '请告诉我您的联系电话。',
        '您好，我需要输入您的联系电话。']

timelist=[
        '请问您需要购买什么时间的机票？',
        '您需要预定哪个时间的机票？',
        '请告诉我您预计的出行时间是？',
        '请问您的出行时间是？',
        '请告知您的出行时间。',
        '您什么时候出发？',
        '您要订哪天的机票？',
        '您要订什么时候的机票？',
        '您想买什么时候？',
        '您什么时候走？',
        '何时出发？',
        '什么时候的？',
        '什么时间的？',
        '好的，请提供出行时间。',
        '时间是？',
        '几号的？',
        '您需要什么时候走？',
        '说一下您的出行时间，谢谢。']





totallist=[]
totallist.extend(namelist)
totallist.extend(countlist)
totallist.extend(departurelist)
totallist.extend(destinationlist)
totallist.extend(timelist)
totallist.extend(idnumberlist)
totallist.extend(phonelist)
totallist.append('已经为您预订完毕。')

# f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa28','r')
# f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa29_0104/qa29','r')
# f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa31_0105/qa31_oneSlot','r')
# f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa33_0125/qa33_0126_23q','r')
f=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa34_0126/qa34_sameOrder_23q','r')
out=f.read()
out=out.split('\r\n')#[:-1]
# ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa28_ticket_randOrder_ANS_slot_test.txt','r')
# ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa29_ticket_randOrder_withSlot_test.txt','r')
# ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa31_noSlot_ticket_rand_withSlot_test.txt','r')
# ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa31_ticket_randOrderAnsSent_withSlot_test.txt','r')
# ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa33_ticket_randAll_merge_test.txt','r')
ff=open('/home/shin/DeepLearning/MemoryNetwork/MemNN/DataCoupus/1230实验数据/qa34_0126/qa34_ticket_sameOrder_merge_test.txt','r')
storys=ff.read().split('\n1 ')
one_story_list=[]
one_story_list.append(storys[0])
for i in range(1,len(storys)):
    one_story_list.append('1 '+storys[i])
assert len(one_story_list)==1000


total_status=0
total_next=0
all_correct=0
slot_correct=0
slot_all_correct=0
total_target=float(998)
total_question=23
for i,one_story in enumerate(one_story_list[:int(total_target)]):
    print i
    this_out=out[i].split('\t')
    status_target=re.findall('status \?\t(\d+?)\n',one_story)
    next_target=re.findall(r'next \?\t(.+?)\t',one_story)
    assert len(status_target)==8
    assert len(next_target)==8
    count_target=re.findall('count \?\t(.+?)\t',one_story)[0]
    name_target=re.findall('name \?\t(.+?)\t',one_story)[0]
    destination_target=re.findall('destination \?\t(.+?)\t',one_story)[0]
    departure_target=re.findall('departure \?\t(.+?)\t',one_story)[0]
    idnumber_target=re.findall('idnumber \?\t(.+?)\t',one_story)[0]
    time_target=re.findall('time \?\t(.+?)\t',one_story)[0]
    phone_target=re.findall('phone \?\t(.+?)\t',one_story)[0]

    slot_in_one=0
    if this_out[16]==count_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[17]==name_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[18]==destination_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[19]==departure_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[20]==idnumber_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[21]==time_target:
        slot_correct+=1
        slot_in_one+=1
    if this_out[22]==phone_target:
        slot_correct+=1
        slot_in_one+=1

    if slot_in_one==7:
        slot_all_correct+=1




    correct=0
    for j in range(8):
        rest_list=['已经为您预订完毕。']
        if this_out[2*j]==status_target[j]:
            total_status+=1

        if status_target[j][0]=='0':
            rest_list.extend(namelist)
        if status_target[j][1]=='0':
            rest_list.extend(countlist)
        if status_target[j][2]=='0':
            rest_list.extend(departurelist)
        if status_target[j][3]=='0':
            rest_list.extend(destinationlist)
        if status_target[j][4]=='0':
            rest_list.extend(timelist)
        if status_target[j][5]=='0':
            rest_list.extend(idnumberlist)
        if status_target[j][6]=='0':
            rest_list.extend(phonelist)


        # print this_out[2*j+1]

        if this_out[2*j+1] in rest_list:
            correct+=1
            if j==6:
                print 'answer:'+this_out[2*j+1]
                print 'target:'+rest_list[1]
                print '\n'

            total_next+=1


        # elif this_out[2*j+1] not in totallist:
        #     print 'The story%d the question %d exists error.'%(i,j)
    if correct==8:
        all_correct+=1

print 'The error of status:%f'%(1-total_status/(8*total_target))
print 'The error of next sentence:%f'%(1-total_next/(8*total_target))
print 'All correct:',all_correct/total_target

print slot_correct
print 'The error of slot:%f'%(1-slot_correct/(total_target*7))
print 'All correct:',slot_all_correct/total_target

pass
