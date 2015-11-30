'''
Created on 2015-11-11

@author: dell
'''

import jieba


FilePath = 'D:/360pan/Data/FieldDialog/copus/Hotel.txt'
destTrainFilePath = 'D:/360pan/Data/FieldDialog/copus/Hotel_Train.txt'
destTestFilePath = 'D:/360pan/Data/FieldDialog/copus/Hotel_Test.txt'

#TicketSlot = {'name', 'idnumber', 'count', 'destination', 'departure', 'time'}
#WeatherSlot = {'area', 'time'}
#RestaurantSlot = {'time', 'destination', 'order_time', 'phone', 'name'}
#RestauSlot = {'order_time', 'name', 'client_count'}
HotelSlot = {'client_phone', 'order_time', 'room_type', 'room_count', 'client_name', 'order_how_long'}
#RechargeSlot = {'price', 'phone'}

currentSlot = HotelSlot

TrainNum = 1000
TestNum = 1000


def segSentence(sentence):
    seg_list = jieba.cut(sentence)
    return ' '.join(seg_list)
     


# fid = open(templateFilePath, 'r')
# lines = fid.readlines()
# templateDict = {}
# for line in lines:
#     [tag, value] = line.strip().split('=')
#     seg_list = jieba.cut(value)
#     value = " ".join(seg_list)[2:]
#     templateDict[tag] = value

fid = open(FilePath, 'r')

fid1 = open(destTrainFilePath, 'w')


docList = []

line = fid.readline()

query = []
story = []
story_index = 0
answer = []
supportIndex = 0 

iter1 = 0
writeTrainFlag = True
storyList = []
while line:
    
    sentence = line.strip()
    
    
    if sentence.find('dialogue') > -1:
        iter1 += 1
        if len(storyList) > 0:
            for i, sentence in enumerate(storyList):
                fid1.write('%d %s\n'%(i+1, sentence))
        # start new story
        storyList = []
        story_index = 1
        
        if iter1 > TrainNum and writeTrainFlag: 
            fid1 = open(destTestFilePath, 'w')
            writeTrainFlag = False
        if iter1 > TrainNum + TestNum: break
    
    elif sentence.find('negate')>-1:
        storyList.append(segSentence(sentence.split('\t')[0][2:])[2:])
        
    elif sentence.find('greeting')>-1 or sentence.find('request')>-1:
        storyList.append(segSentence(sentence.split('\t')[0][2:])[2:])
             
    
    elif sentence.find('inform')>-1:
        #print sentence.split('\t')[0]
        #print sentence.split('\t')[0][2:]
        #print segSentence(sentence.split('\t')[0][2:])[2:]
        storyList.append(segSentence(sentence.split('\t')[0][2:])[2:])
        
        answer = sentence.split('\t')[1][7:]
        if answer.find('inform') > -1:
            answer = answer.replace('inform', '+')
        elif answer.find(',') > -1:
            answer = answer.replace(',', '+')
        answer_List = []
        if answer.find('+') > -1:
            answer_List = answer.split('+')
        else:
            answer_List.append(answer)
        
        supportIndex = len(storyList)
        slotList = []
        for anw in answer_List:
            answer = anw[anw.find('=')+1:].strip(' ')
            answer = answer.replace(" ", '')
            tag = anw[0:anw.find('=')].strip(' ') + " ?"
            slotList.append(anw[0:anw.find('=')].strip(' '))
            storyList.append("%s\t%s\t%d"%(tag, ','.join(jieba.cut(answer)), supportIndex))
        
        nilSlot = currentSlot - set(slotList)
        for tag in nilSlot:
            storyList.append("%s\t%s\t%d"%(tag + ' ?', 'nil', supportIndex))   
    line = fid.readline()




fid.close()
fid1.close()

print 'Done'     
