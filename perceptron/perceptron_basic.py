import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

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

    # visualize the dataset
    x = data[:,0]
    y = data[:,1]
    flag = data[:,2]
    size = 50*np.ones(flag.shape)
    # open the interactive mode
    plt.ion()
    plt.show()
    # initialize the weights and bias
    w = np.random.rand(2)
    b = np.random.rand(1)
    # set the learning rate
    alpha = 0.01
    plt.plot()
    for epoch in range(100):
        pred = (np.sign(data[:,0:2]@w+b)).squeeze()
        mask = (pred!=data[:,2])
        mask = np.where(mask==1)
        mask = np.array(mask,dtype=int).squeeze()
        # draw the seperate line and the points
        plt.cla()
        error_points = np.ma.masked_where(flag!=pred,size)
        right_points = np.ma.masked_where(flag==pred,size)
        plt.scatter(x,y,s=error_points,c='b')
        plt.scatter(x,y,s=right_points,c='r')
        plt.plot((-(b/w[1]),0),(0,-(b/w[0])))
        plt.title('epoch={} weight={} bias={}'.format(epoch,w,b), 
                    fontdict={'size': 10, 'color':  'm'})            
        plt.pause(0.1)
        if mask.size!=0:
            # update the weight
            random_index = np.random.choice(mask)
            gradient = data[random_index,2]*data[random_index,:2]
            w = w + 0.1*gradient
            b = b + 0.1*data[random_index,2]
        # visualize the data
        else:
            break
    plt.ioff()
    plt.show()

        
        





