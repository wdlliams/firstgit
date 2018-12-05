# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#代码中带“#”标注的语句是和原始gan相比多出来的，其他部分和原始gan一样
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('../../MNIST_data',one_hot=True)
mb_size=64
Z_dim=100
X_dim=mnist.train.images.shape[1]
y_dim=mnist.train.labels.shape[1]
h_dim=128
def xavier_init(size):
    in_dim=size[0]
    xavier_stddev=1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=size,stddev=xavier_stddev)
X=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,y_dim]) #和原始gan相比多余的condition y
D_W1=tf.Variable(xavier_init([X_dim+y_dim,h_dim])) #和原始gan相比多余的condition y
D_b1=tf.Variable(tf.zeros(shape=[h_dim]))
D_W2=tf.Variable(xavier_init([h_dim,1]))
D_b2=tf.Variable(tf.zeros(shape=[1]))
theta_D=[D_W1,D_b1,D_W2,D_b2]
def discriminator(x,y): #和原始gan相比多余的condition y
    inputs=tf.concat(axis=1,values=[x,y]) #和原始gan相比多余的condition y
    D_h1=tf.nn.relu(tf.matmul(inputs,D_W1)+D_b1)
    D_logit=tf.matmul(D_h1,D_W2)+D_b2
    D_prob=tf.nn.sigmoid(D_logit)
    return D_prob,D_logit
Z=tf.placeholder(tf.float32,[None,Z_dim])
G_W1=tf.Variable(xavier_init([Z_dim+y_dim,h_dim])) #和原始gan相比多余的condition y
G_b1=tf.Variable(tf.zeros(shape=[h_dim]))
G_W2=tf.Variable(xavier_init([h_dim,X_dim]))
G_b2=tf.Variable(tf.zeros(shape=[X_dim]))
theta_G=[G_W1,G_b1,G_W2,G_b2]
def generator(z,y): #和原始gan相比多余的condition y
    inputs=tf.concat(axis=1,values=[z,y]) #和原始gan相比多余的condition y
    G_h1=tf.nn.relu(tf.matmul(inputs,G_W1)+G_b1)
    G_log_prob=tf.matmul(G_h1,G_W2)+G_b2
    G_prob=tf.nn.sigmoid(G_log_prob)
    return G_prob
def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])
G_sample=generator(Z,y)
D_real,D_logit_real=discriminator(X,y)
D_fake,D_logit_fake=discriminator(G_sample,y)
D_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,labels=tf.ones_like(D_logit_real)))
D_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.zeros_like(D_logit_fake)))
D_loss=D_loss_real+D_loss_fake
G_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.ones_like(D_logit_fake)))
D_solver=tf.train.AdamOptimizer().minimize(D_loss,var_list=theta_D)
G_solver=tf.train.AdamOptimizer().minimize(G_loss,var_list=theta_G)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
i=0
for i in range(100000):
    X_mb,y_mb=mnist.train.next_batch(mb_size)
    Z_sample=sample_Z(mb_size,Z_dim)
    _,D_loss_curr=sess.run([D_solver,D_loss],feed_dict={X:X_mb,y:y_mb,Z:Z_sample}) #和原始gan相比多余的condition y
    _,G_loss_curr=sess.run([G_solver,G_loss],feed_dict={y:y_mb,Z:Z_sample}) #和原始gan相比多余的condition y
    i+=1
    if i%1000==0:
        print('Iter:{}'.format(i))
        print('D loss:{:.4}'.format(D_loss_curr))
        print('G loss:{:.4}'.format(G_loss_curr))
        print()