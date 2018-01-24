import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(r"C:\Users\mashiji\Desktop\MNIST",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)

# (55000, 784) (55000, 10)
# (5000, 784) (5000, 10)
# (10000, 784) (10000, 10)

print(mnist.train.labels[2])

x = tf.placeholder(tf.float32,[None, 784]) #图像输入向量  
W = tf.Variable(tf.zeros([784,10]))  #权重，初始化值为全零  
b = tf.Variable(tf.zeros([10]))  #偏置，初始化值为全零  
  
#进行模型计算，y是预测，y_ 是实际  
y = tf.nn.softmax(tf.matmul(x,W) + b)  
  
y_ = tf.placeholder("float", [None,10])  
  
#计算交叉熵  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
#接下来使用BP算法来进行微调,以0.01的学习速率  
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
  
#上面设置好了模型，添加初始化创建变量的操作  
init = tf.global_variables_initializer()  
#启动创建的模型，并初始化变量  
sess = tf.Session()  
sess.run(init)  
#开始训练模型，循环训练1000次  
for i in range(1000):  
    #随机抓取训练数据中的100个批处理数据点  
    batch_xs, batch_ys = mnist.train.next_batch(100)  
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})  
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))  
#''''' 进行模型评估 '''  
  
#判断预测标签和实际标签是否匹配  
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
#计算所学习到的模型在测试数据集上面的正确率  
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})) 
