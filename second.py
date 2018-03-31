import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    print(weights.shape)
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    print(biases.shape)
    wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs=wx_plus_b
    else:
        outputs=activation_function(wx_plus_b)
    return outputs
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.RMSPropOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i %50==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,'r',lw=5)
        plt.pause(0.5)
plt.show()
