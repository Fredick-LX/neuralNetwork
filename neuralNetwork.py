# -*- encoding: gbk -*-
import os
try:
    import numpy
    import scipy.special
except:
    print("����δ��װnumpy���scipy��")
    print("��Ϊ���Զ���װ\n")
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
            print("Ȩ�ؾ���wih�Ѷ�ȡ���......")
        except:
            self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            print("Ȩ�ؾ���wih�����ɹ�")
        try:
            who1 = open("who")
            who2 = who1.read()
            who3 = who2.split(",")
            for i in range(self.hnodes*self.onodes):
                who3[i] = float(who3[i])
            a = numpy.array(who3)
            who3 = a.reshape(self.onodes,self.hnodes)
            self.who = who3
            print("Ȩ�ؾ���who�Ѷ�ȡ���......\n")
        except:
            self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
            print("Ȩ�ؾ���who�����ɹ�\n")
        print("����n.save()�Ա���Ȩ�ؾ���[����ȷ��Ȩ�غ��ʺ󱣴�\n�󱣴�ֻ���ҵ�Դ�ļ�,��Ȩ�ؾ���ɾ��]\n����n.show()�Բ鿴Ȩ�ؾ���\n����main_train(tl)��ѵ��������\n����test()�Բ���������\n")
        self.lr = learningrate       
        self.activation_function = lambda x: scipy.special.expit(x)     
        pass
    
    def train(self, inputs_list, targets_list):
        #ѵ��
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
        #��ѯ
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def save(self):
        #����Ȩ�ؾ���
        f1 = open("wih", 'w+')  
        for i in range(self.hnodes):
            for j in range(self.inodes):
                r = self.wih[i,j]
                if i == self.hnodes-1 and j == self.inodes-1:
                    print(r,file=f1,end="")
                else:
                    print(r,file=f1,end=",")
        print("Ȩ�ؾ���wih�ѱ������......")
        f2 = open("who", 'w+')  
        for k in range(self.onodes):
            for l in range(self.hnodes):
                r = self.who[k,l]
                if k == self.onodes-1 and l == self.hnodes-1:
                    print(r,file=f2,end="")
                else:
                    print(r,file=f2,end=",")
        print("Ȩ�ؾ���who�ѱ������......\n����Ȩ�ؾ����ѱ������")
        
    def show(self):
        print("����㵽���ز��Ȩ�ؾ���Ϊ:\n",self.wih)
        print("���ز㵽������Ȩ�ؾ���Ϊ:\n",self.who)


n = neuralNetwork(784,100,10,0.2)

def test():
    #�Բ������ݽ��в�ѯ
        Grayer.test()
        alls2 = Grayer.iner
        global b
        b = [i/255.0*0.99+0.01 for i in alls2]
        print("��������Ϊ��д��������:")
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

if input("Ҫѵ����?") in "yY��":
    print("��ʼѵ��>>>")
    main_train(tl)
    print("ѵ�����...")
if input("Ҫ������?") in "yY��":
    print("��ʼ����>>>")
    main_test(sl)
    print("�������...")
