'''
Created on 2015-11-23

@author: dell
'''
import re

class TimeString(object):
    start= 0
    end = 0
    value = ''
    tag = ''
    def __init__(self, start, end, value, tag):
        self.start =start
        self.end = end
        self.value = value
        self.tag =  tag


class TimeRecognition(object):
    rulePath = ''
    regexRules = {}
    def __init__(self, rulePath):
        regexFid = open(rulePath, 'r')
        self.regexRules = {}
        for line in regexFid.readlines():
            if line[0] is not '#':
                line = unicode(line, 'utf-8')
                [rules, tag] = line.strip().split('\t')
                self.regexRules[rules] = tag
    
        print 'Finish TimeRecog Initialization'
    
    def ExtracTimeUnit(self, sentence):
        result = []
        if not isinstance(sentence,unicode):
                sentence=unicode(sentence,'utf8')
        for rule, tag in self.regexRules.iteritems():

            m = re.search(rule, sentence)
            if m:
                #print m.start(), m.end(), sentence[m.start():m.end()], tag
                timeString = TimeString(m.start(), m.end(), sentence[m.start():m.end()], tag)
                result.append(timeString)
        
        return result
    
    def process(self, sentence):
        if not isinstance(sentence,unicode):
            sentence=unicode(sentence,'utf8')
        TimeUnit = self.ExtracTimeUnit(sentence)
        MergeTimeUnit = self.merge(TimeUnit)
        AfterDelTimeUnit = self.delete(MergeTimeUnit)
        result = self.replace(sentence, AfterDelTimeUnit)
    
        return result
    
    
    
    def merge(self, TimeUnit):
        if len(TimeUnit) == 0:
            return []
        TimeUnit = sorted(TimeUnit, cmp=lambda x,y:cmp(x.start,y.start),  reverse=False)
        result = []
        
        LastMerge = False
        for i in range(len(TimeUnit)-1):
            start = i
            value = TimeUnit[i].value
            while TimeUnit[i].end == TimeUnit[i+1].start - 1:
                value += TimeUnit[i+1].value
                i += 1
                if i == len(TimeUnit)-1:
                    LastMerge = True
                    break
                
            newTimeString = TimeString(TimeUnit[start].start, TimeUnit[i].end, value, TimeUnit[start].tag)
            result.append(newTimeString)
        if not LastMerge: result.append(TimeUnit[-1]) 
        return result
        
    
    
    def delete(self, TimeUnit):
        if len(TimeUnit) == 0: return []
        
        resultSet = set(TimeUnit)
        for cmp1 in range(len(TimeUnit)-1):
            for cmp2 in range(cmp1+1, len(TimeUnit)):
                e1 = TimeUnit[cmp1]
                e2 = TimeUnit[cmp2]
                if (e1.start >= e2.start and e1.end <= e2.end):
                    resultSet.remove(e1)
                elif (e1.start <= e1.start and e1.end >= e2.end):
                    resultSet.remove(e2)
                    
        result = list(resultSet)
        
        return result
                   
                               
    def replace(self, sentence, TimeUnits):
        for TimeUnit in TimeUnits:
            sentence = ('%s%s%s'%(sentence[0:TimeUnit.start], TimeUnit.value, sentence[TimeUnit.end:]))
        
        return sentence

        
        
     
                
    
    