import numpy as np
import matplotlib.pyplot as plt

def create_dataset(num):
    # create a dataset randomly
    pos = np.ones((num,2))
    neg = np.ones((num,2))
    pos_target = np.ones((num,1))
    neg_target = np.ones((num,1))
    neg_target.fill(-1)
    pos = np.random.normal(3*pos,0.5)
    neg = np.random.normal(neg,0.5)
    pos = np.concatenate((pos,pos_target),axis=1)
    neg = np.concatenate((neg,neg_target),axis=1)

    dataset = np.concatenate((pos,neg),axis=0)
    np.random.shuffle(dataset)      
    return dataset

if __name__ == '__main__':
    data = create_dataset(20)
    points = data[:,0:2]
    flag = data[:,2]
    x = points[:,0]
    y = points[:,1]
    size = 50*np.ones(flag.shape)
    # calculate the gram matrix
    gram = np.zeros((points.shape[0],points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            gram[i][j] = np.dot(points[i],points[j])

    # initialize the weights and bias
    alpha = np.zeros((40))
    beta = 0
    # set the learning rate
    lr = 0.3
    # open the interactive mode
    plt.ion()
    plt.show()
    random_index = None
    for epoch in range(100):
        pred = np.sign(gram@(flag*alpha)+beta)
        mask = (pred!=data[:,2])
        mask = np.where(mask==1)
        mask = np.array(mask,dtype=int).squeeze()
        plt.cla()
        error_points = np.ma.masked_where(flag!=pred,size)
        right_points = np.ma.masked_where(flag==pred,size)
        blue = plt.scatter(x,y,s=error_points,c='b')
        red = plt.scatter(x,y,s=right_points,c='r')          
        plt.title('epoch={} random={} beta={}'.format(epoch,random_index,beta), 
            fontdict={'size': 10, 'color':  'm'})
        plt.legend([red,blue],['Misclassification','Corret classification']) 
        plt.pause(0.1) 
        if mask.size!=0:
            random_index = np.random.choice(mask)
            alpha[random_index] = alpha[random_index] + lr
            beta = beta + lr*flag[random_index]

        # visualize the data
        else:
            break
    plt.ioff()
    plt.show()