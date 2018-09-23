__author__ = 'jmh081701'
import numpy
import  random
class IrisData(object):
    def __init__(self,datapath="bezdekIris.data.txt",seperate_rate = 0.1):
        x = []
        y = []
        with open(datapath) as fp:
            lines = fp.readlines()
            for each in lines:
                each= each.replace("\n",'')
                _da= each.split(',')
                _x=[]
                for i in range(len(_da)):
                    if(i==(len(_da)-1)):
                        if _da[i]=='Iris-setosa':
                            y.append([1.0,0.,0.])
                        if _da[i]=='Iris-versicolor':
                            y.append([0.,1.,0.])
                        if _da[i]=='Iris-virginica':
                            y.append([0.,0.,1.0])
                        continue
                    _x.append(float(_da[i]))
                x.append(_x)

        train_index =set()
        test_index=set()
        test_number =int( len(x) * seperate_rate)

        while len(test_index) < test_number:
            test_index.add(random.randint(0,len(x)-1))
        for i in range(len(x)):
            if i not in test_index:
                train_index.add(i)
        self.x = self.normalize(x)
        self.y = y
        self.train_index = train_index
        self.test_index = test_index
    def normalize(self,x):
        _x = numpy.array(x)
        _min=numpy.min(_x,0)
        _max=numpy.max(_x,0)
        _x =(_x-_min)/((_max-_min))
        return _x
    def train(self):
        train_x=[]
        train_y=[]
        for i in self.train_index:
            train_x.append(self.x[i])
            train_y.append(self.y[i])
        return  numpy.array(train_x),numpy.array(train_y)
    def test(self):
        test_x=[]
        test_y=[]
        for i in self.test_index:
            test_x.append(self.x[i])
            test_y.append(self.y[i])
        return  numpy.array(test_x),numpy.array(test_y)
    def next_train(self,batch_size=40):
        _x = []
        _y =[]
        train_index =list(self.train_index)
        while len(_x) < min(batch_size,len(train_index)):
            index = random.randint(0,len(train_index)-1)
            _x.append(self.x[train_index[index]])
            _y.append(self.y[train_index[index]])
        return  numpy.array(_x),numpy.array(_y)
    def next_test(self,batch_size=40):
        _x = []
        _y =[]
        test_index =list(self.test_index)
        while len(_x) < min(batch_size,len(test_index)):
            index = random.randint(0,len(test_index)-1)
            _x.append(self.x[test_index[index]])
            _y.append(self.y[test_index[index]])
        return  numpy.array(_x),numpy.array(_y)