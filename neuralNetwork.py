# -*- encoding: gbk -*-
import os
try:
    import numpy
    import scipy.special
except:
    print("可能未安装numpy库或scipy库")
    print("将为您自动安装\n")
    os.system("pip install numpy")
    os.system("pip install scipy")
import Grayer

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        try:
            wih1 = open("wih")
            wih2 = wih1.read()
            wih3 = wih2.split(",")
            for i in range(self.inodes*self.hnodes):
                wih3[i] = float(wih3[i])
            a = numpy.array(wih3)
            wih3 = a.reshape(self.hnodes,self.inodes)
            self.wih = wih3
            print("权重矩阵wih已读取完成......")
        except:
            self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            print("权重矩阵wih创建成功")
        try:
            who1 = open("who")
            who2 = who1.read()
            who3 = who2.split(",")
            for i in range(self.hnodes*self.onodes):
                who3[i] = float(who3[i])
            a = numpy.array(who3)
            who3 = a.reshape(self.onodes,self.hnodes)
            self.who = who3
            print("权重矩阵who已读取完成......\n")
        except:
            self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
            print("权重矩阵who创建成功\n")
        print("输入n.save()以保存权重矩阵[请在确定权重合适后保存\n误保存只能找到源文件,将权重矩阵删除]\n输入n.show()以查看权重矩阵\n输入main_train(tl)以训练神经网络\n输入test()以测试神经网络\n")
        self.lr = learningrate       
        self.activation_function = lambda x: scipy.special.expit(x)     
        pass
    
    def train(self, inputs_list, targets_list):
        #训练
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T 
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        self.who += self.lr *numpy.dot((output_errors * final_outputs *
                    (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr *numpy.dot((hidden_errors * hidden_outputs *
                   (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        #查询
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def save(self):
        #保存权重矩阵
        f1 = open("wih", 'w+')  
        for i in range(self.hnodes):
            for j in range(self.inodes):
                r = self.wih[i,j]
                if i == self.hnodes-1 and j == self.inodes-1:
                    print(r,file=f1,end="")
                else:
                    print(r,file=f1,end=",")
        print("权重矩阵wih已保存完成......")
        f2 = open("who", 'w+')  
        for k in range(self.onodes):
            for l in range(self.hnodes):
                r = self.who[k,l]
                if k == self.onodes-1 and l == self.hnodes-1:
                    print(r,file=f2,end="")
                else:
                    print(r,file=f2,end=",")
        print("权重矩阵who已保存完成......\n所有权重矩阵已保存完成")
        
    def show(self):
        print("输入层到隐藏层的权重矩阵为:\n",self.wih)
        print("隐藏层到输出层的权重矩阵为:\n",self.who)


n = neuralNetwork(784,100,10,0.2)

def test():
    #对测试数据进行查询
        Grayer.test()
        alls2 = Grayer.iner
        global b
        b = [i/255.0*0.99+0.01 for i in alls2]
        print("神经网络认为您写的数字是:")
        print(numpy.argmax(n.query(b)))
        Grayer.iner.clear()

def main_train(training_data_list):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets = numpy.zeros(10)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
def main_test(test_data_list):
    s = []
    for record in test_data_list:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        if (label == correct_label):
            s.append(1)
        else:
            s.append(0)
    print(s)
    s_a = numpy.asarray(s)
    print(s_a.sum()/s_a.size)
tf = open("mnist_train.csv","r")
tl = tf.readlines()
tf.close()
sf = open("mnist_test.csv","r")
sl = sf.readlines()
sf.close()

if input("要训练吗?") in "yY是":
    print("开始训练>>>")
    main_train(tl)
    print("训练完成...")
if input("要测试吗?") in "yY是":
    print("开始测试>>>")
    main_test(sl)
    print("测试完成...")
