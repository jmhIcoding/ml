#coding:utf8
__author__ = 'jmh081701'
import tensorflow as tf
from tool.genData  import IrisData as iris
data = iris()
n=4
m=3
hidden_layer=[8]
input_x = tf.placeholder(dtype=tf.float32,shape=[None,n],name="input_matrix")
input_y = tf.placeholder(dtype=tf.float32,shape=[None,m],name="expect_output_matrix")
all_layer=[n]+hidden_layer+[m] #所有的网络层,包括输入层、隐藏层、输出层
W=[]#所有的权值矩阵参数，W[i]表示第i层与前一层的权值，其中W
b=[]#所有的偏置参数
activate_func=tf.nn.sigmoid
def gen_parameter(all_layer):
    W=[]
    b=[]
    for i in range(len(all_layer)):
        W.append(0)
        b.append(0)
    for i in range(1,len(all_layer)):
        W[i]=tf.Variable(initial_value=tf.truncated_normal(shape=(all_layer[i-1],all_layer[i]),stddev=0.1),name="W%dto%d"%(i-1,i))
        b[i]=tf.zeros(shape=(1,all_layer[i]),name="bias%d"%i)
    return W,b
W,b= gen_parameter(all_layer)
a=[]
z=[]
for i in range(len(all_layer)):
        a.append(0)
        z.append(0)
a[0]=input_x
z[0]=0
#for i in range(1,len(all_layer)):
#        z[i]=tf.add(tf.matmul(a[i-1],W[i]),b[i])
#        a[i]=activate_func(z[i])
z[1]=tf.matmul(a[0],W[1])+b[1]
a[1]=activate_func(z[1])

z[2]=tf.matmul(a[1],W[2])+b[2]
a[2]=activate_func(z[2])

predict_y = tf.nn.softmax(a[2],name="predict_y")
predict_label = tf.arg_max(predict_y,1)
real_lable = tf.arg_max(input_y,1)
error_rate = 1-tf.to_float(tf.reduce_sum(tf.to_int32(tf.equal(predict_label,real_lable))))/tf.to_float(tf.shape(real_lable)[0])
loss= tf.reduce_sum(-input_y*tf.log(predict_y))
trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    cnt =0
    while True:
        cnt+=1
        train_x,train_y = data.next_train(batch_size=100)
        _predicty,_loss,_trainer=sess.run(fetches=[predict_y,loss,trainer],feed_dict = {input_x:train_x,input_y:train_y})
        if cnt % 1000==0:
            print("valid......")
            test_x,test_y= data.next_test()
            _error_rate,_predict,_predictl,_rl, = sess.run([error_rate,predict_y,predict_label,real_lable],feed_dict ={input_x:test_x,input_y:test_y})
            print(_error_rate)
            print(_predictl,_rl)
            if cnt >30000 :
                print("Over!")
                break
        if cnt % 100 ==0:
            print({"cnt":cnt,"loss":_loss})
    test_x,test_y= data.test()
    _error_rate,_predict,_predictl,_rl, = sess.run([error_rate,predict_y,predict_label,real_lable],feed_dict ={input_x:test_x,input_y:test_y})
    print(_error_rate)
    print(_predictl,_rl)

