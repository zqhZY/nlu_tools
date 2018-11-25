# -*- coding: utf-8 -*-
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# All labels
testClasses = ['website', 'tvchannel', 'lottery', 'chat', 'match',
          'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
          'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
          'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
          'flight', 'translation', 'train', 'schedule', 'radio', 'email']

'''
    Caculate the accuracy ratio, recall ratio, precision ratio and F1 value
    Json file format: testcases{"id": {
        "query": "",
        "label": "",
        }
    }
'''
def caculateF(real_dict, pred_dict):
    confus_mat = {}
    testcase_nums = {}

    for label in testClasses:
        confus_mat[label] = {}
        testcase_nums[label] = 0
        for it in testClasses:
            confus_mat[label][it] = 0

    for id in real_dict:
        true_label = real_dict[id]['label']
        pred_label = pred_dict[id]['label']
        if true_label in confus_mat and pred_label in confus_mat:
            confus_mat[pred_label][true_label] += 1
            testcase_nums[pred_label] += 1

    correctTotalNum = sum([confus_mat[tClass][tClass] for tClass in testClasses])
    if sum(testcase_nums.values()) == 0:
        accuracy = 0
    else:
        accuracy = float(correctTotalNum) / float(sum(testcase_nums.values()))


    # Precision
    precisionDict = {}
    for tClass in testClasses:
        testPos = sum([confus_mat[c][tClass] for c in testClasses])
        truePos = confus_mat[tClass][tClass]
        if testPos == 0:
            precisionDict[tClass] = 0.0
        else:
            precisionDict[tClass] = float(truePos) / float(testPos)
    assert len(precisionDict) == len(testClasses)

    # Recall
    recallDict = {}
    for tClass in testClasses:
        truePos = confus_mat[tClass][tClass]
        if testcase_nums[tClass] == 0:
            print "{} has no case!".format(tClass)
            exit()
        recallDict[tClass] = float(truePos) / float(testcase_nums[tClass])
    assert len(recallDict) == len(testClasses)

    # F1 score
    FDict = {}
    for tClass in testClasses:
        if precisionDict[tClass] + recallDict[tClass] == 0:
            FDict[tClass] = 0.0
        else:
            FDict[tClass] = float(2 * precisionDict[tClass] * recallDict[tClass]) / \
                float(precisionDict[tClass] + recallDict[tClass])
    assert len(FDict) == len(testClasses)

    # avg F1 score
    aveF = sum(FDict.values()) / float(len(FDict))

    return accuracy, precisionDict, recallDict, FDict, aveF

if __name__ =='__main__':
    if len(sys.argv) < 3:
        print 'Too few args for this script'
        exit(1)
    truth_dict = json.load(open(sys.argv[1]), encoding='utf8')
    pred_dict = json.load(open(sys.argv[2]), encoding='utf8')
    acc, _, _, _, Fscore = caculateF(truth_dict, pred_dict)
    print 'Avg F1 Score: %f' % Fscore