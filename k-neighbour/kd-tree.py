import numpy as np
import os
import operator


def Dis(tree,inX,list_node=[]):
    if tree == None:
        return list_node

    if tree.left==None and tree.right==None:
        dis=tree.calculateDistance(inX)
        list_node.append([dis,tree.node[-1]])
        return list_node
    else:
        dim = tree.depth%3
        if inX[dim]>tree.node[dim]:
            dis=tree.calculateDistance(inX)
            list_node.append([dis,tree.node[-1]])
            list_node = Dis(tree.right, inX, list_node)
        else:
            dis=tree.calculateDistance(inX)
            list_node.append([dis,tree.node[-1]])
            list_node = Dis(tree.left, inX, list_node)
    return list_node

def classify(inX, tree, k):
    '''
    classify
    '''
    dis_list = Dis(tree,inX,[])
    k_list = (sorted(dis_list,key=lambda x:x[0]))[0:k]
    k_dict={'1':0,'2':0,'3':0}
    for item in k_list:
        k_dict[str(int(item[1]))]+=1
    predict = max(zip(k_dict.values(),k_dict.keys()))
    return int(predict[1])


class KdTree():
    def __init__(self, root):
        self.node = root #numpy array
        self.left = None
        self.right = None
        self.depth = None
    def addLeft(self,leftchild):
        self.left = leftchild
    def addRight(self, rightchild):
        self.right = rightchild
    def calculateDistance(self,node):
        return np.sum((self.node[0:3]-node)**2)

def traverse(root):           
    if root==None:  
        return  
    print(root.node[0])  
    traverse(root.left)  
    traverse(root.right)

def buildKdTree(dataSet, depth=0):
    if dataSet.size == 0:
        return None
    if depth == 0:
        dataSet=np.asarray(sorted(dataSet,key=lambda x: x[0],reverse=False))
        index = int(len(dataSet)/2)
        median = dataSet[index]
        kdTree = KdTree(median)
        kdTree.depth=0
        dataSetL = dataSet[0:index]
        dataSetR = dataSet[index+1:]
        kdTree.addLeft(buildKdTree(dataSetL,depth=1))
        kdTree.addRight(buildKdTree(dataSetR,depth=1))
        return kdTree
    else:
        dim = depth%3
        dataSet=np.asarray(sorted(dataSet,key=lambda x: x[dim],reverse=False))
        index = int(len(dataSet)/2)
        median = dataSet[index]
        kdTree = KdTree(median)
        kdTree.depth=depth
        dataSetL = dataSet[0:index]
        dataSetR = dataSet[index+1:]
        kdTree.addLeft(buildKdTree(dataSetL,depth+1))
        kdTree.addRight(buildKdTree(dataSetR,depth+1))
        return kdTree

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
        classLabelVector.append(listFromLine[-1])
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
    normMat = np.asarray(normMat)
    labels = np.asarray(labels)
    labels = np.expand_dims(labels,axis=1)
    mat = np.concatenate((normMat,labels),axis=1).astype(np.float)
    tree = buildKdTree(mat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyresult = classify((inArr - minVals)/ranges,tree, 20)
    print(" You will probably like the person:",resultList[int(classifyresult) - 1])

if __name__ == '__main__':
    classifyPerson()
   