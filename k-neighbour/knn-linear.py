import numpy as np
import os
import operator


def classify0(inX, dataSet, labels, k):
    # the number of entries
    dataSetSize = dataSet.shape[0]
    '''
    # numpy.tile(a, scale) construct an array by repeating the a #times given by scale
    if a = ([1,2,3])
    b = np.tile(a,2)
    b = ([1,2,3,1,2,3])
    c = np.tile(a,(2,1))
    c = ([1,2,3],
        [1,2,3])
    '''
    # let the inX has the same entries as dataset, calculate the distance, like broadcasting 
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # sort the distances between inX and all points in dataSet
    sortedDistIndicies = distances.argsort()
    # initilize the dict
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # sort the ClassCount by the second value
    sortedClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    # load file, get properties of the file
    fr = open(filename)
    arrayOlines = fr.readlines()
    numerOfLines = len(arrayOlines)
    # intialize the mat for return
    returnMat = np.zeros((numerOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # 消除前后空格
        line = line.strip()
        # 按照split中的参数关键词进行切片
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append((listFromLine[-1]))
        index+=1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    # 0 for cols and 1 for rows
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = dataSet / np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent filer miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, labels = file2matrix('k-neighbour/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyresult = classify0((inArr - minVals)/ranges, normMat, labels, 3)
    print(" You will probably like the person:",resultList[int(classifyresult) - 1])

if __name__ == '__main__':
    classifyPerson()
   