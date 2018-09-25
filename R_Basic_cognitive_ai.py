# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:01:59 2018

@author: Ravi Kamble
"""

#import tensorflow as tf
#
#a = tf.constant([20])
#b = tf.constant([12])
#
#c = tf.add(a,b)
#
#with tf.Session() as sess:
#    result = sess.run(c)
#    print(result)

#Scalar = tf.constant([2])
#Vector = tf.constant([5,6,2])
#Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
#Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
#with tf.Session() as session:
#    result = session.run(Scalar)
#    print("Scalar (1 entry):\n %s \n" % result)
#    result = session.run(Vector)
#    print("Vector (3 entries) :\n %s \n" % result)
#    result = session.run(Matrix)
#    print("Matrix (3x3 entries):\n %s \n" % result)
#    result = session.run(Tensor)
#    print("Tensor (3x3x3 entries) :\n %s \n" % result)
#
#
#Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
#Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])
#
#first_operation = tf.add(Matrix_one, Matrix_two)
#second_operation = Matrix_one * Matrix_two
#
#with tf.Session() as session:
#    result = session.run(first_operation)
#    print("Defined using tensorflow function :")
#    print(result)
#    result = session.run(second_operation)
#    print("Defined using normal expressions :")
#    print(result)

import tensorflow  

  
#Trainable Parameters  
W = tensorflow.Variable([0.3], dtype=tensorflow.float32)  
b = tensorflow.Variable([-0.2], dtype=tensorflow.float32)  

  #Training Data (inputs/outputs)  
x = tensorflow.placeholder(dtype=tensorflow.float32)  
y = tensorflow.placeholder(dtype=tensorflow.float32)  
   
x_train = [1, 2, 3, 4]  
y_train = [0, 1, 2, 3]  
   
 #Linear Model  
linear_model = W * x + b  
   
 #Linear Regression Loss Function - sum of the squares  
squared_deltas = tensorflow.square(linear_model - y_train)  
loss = tensorflow.reduce_sum(squared_deltas)  
   
 #Gradient descent optimizer  
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)  
train = optimizer.minimize(loss=loss)  
   
 #Creating a session 
sess = tensorflow.Session()  
   
writer = tensorflow.summary.FileWriter("/tmp/log/", sess.graph)  
   
 #Initializing variables  
init = tensorflow.global_variables_initializer()  
sess.run(init)  
   
 #Optimizing the parameters  
for i in range(1000):
    sess.run(train, feed_dict={x: x_train, y: y_train})  
   
     #Print the parameters and loss  
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})  
    print("W : ", curr_W, ", b : ", curr_b, ", loss : ", curr_loss)  
       
#    writer.close()  
       
    sess.close()  

    




















