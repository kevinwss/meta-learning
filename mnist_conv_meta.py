# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy as np

import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# step for variable learning rate
step = tf.placeholder(tf.int32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step, the learning rate is a placeholder
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#------------------------

def get_activates(feature_map, top_n):
    activates = []
    for feature in feature_map:   #feature [0.0,0.1,...0.3]
        activates_idx = []
        sort_f = sorted(feature)
        threshold = sort_f[-top_n]
		
        for i in range(len(feature)):
            if feature[i] >= threshold:
                activates_idx.append(i)
        activates.append(activates_idx)
	    
	
    return np.array(activates)  #[[4,10,...500],[]]

def get_label(y):
	labels = []
	for this_y in y:
		for i in range(10):
			if this_y[i] == 1:
				this_label = i
				break
		labels.append(this_label)
	return labels	   #[3,6,6,6...]
		
def compare(activates):
	for i in range(20):
		this_set = set(activates[i] + activates[i+1])
		print(str(i)+" "+str(i+1)+" in common",120-len(this_set))
	'''		
	set01 = set(activates[0] + activates[1])
	set12 = set(activates[1] + activates[2])
	set23 = set(activates[2] + activates[3])
	set34 = set(activates[3] + activates[4])
	
	print("01 in common",120 - len(set01))
	print("12 in common",120 - len(set12))
	print("23 in common",120 - len(set23))
	print("34 in common",120 - len(set34))
	'''
	
def get_common(labels,activates):
	label_summary = [[],[],[],[],[],[],[],[],[],[]]  # classification #[[which index in train set is 0]]
	common_neuron_summary = []
	for i in range(len(labels)):
		label_summary[labels[i]].append(i)
		
	print("label_summary)",label_summary)
	
	for j in range(10): # there are 10 classes

		activate_set = [activates[idx] for idx in label_summary[j]]
		if len(activate_set) == 0:
			common_neuron_summary.append([])
			continue
			
		common_neuron = activate_set[0]
		for activate in activate_set[1:]:
			
			to_delete = []
			for neuron_idx in range(len(common_neuron)):   #activate [#neuron,...]
				neuron = common_neuron[neuron_idx]
				if not (neuron in activate): #record 
					to_delete.append(neuron)    #record neuron to delete
			
			for to_delete_neuron in to_delete:
				common_neuron.remove(to_delete_neuron)
				
		common_neuron_summary.append(common_neuron)
	print("common_neuron_summary" , common_neuron_summary)
	return common_neuron_summary

def predict(learned_activates,activates):
	predicted = []
	ratios = []
	
	for test_pattern in activates: #test_pattern [30,...500]
		activate_ratio_record = []
		for pattern in learned_activates:   #pattern [412,496,...] which stands for 0
			pattern_length = len(pattern)
			common_activate_count = 0
			for neuron in  pattern:
				if neuron in test_pattern:
					common_activate_count += 1
					
			# record the activate ratio		
			if pattern_length == 0:        # this pattern was not learned beforehand
				activate_ratio_record.append(0)
			else:
				activate_ratio_record.append(float(common_activate_count/pattern_length))
				
		# find out the highest ratio
		highest_ratio = max(activate_ratio_record)
		ratios.append(highest_ratio)
		predicted_class = activate_ratio_record.index(highest_ratio)
			
		predicted.append(predicted_class)
	
	print("ratios",ratios)	
	return predicted
	
def get_score(gt,predict):
	correct_count = 0
	total = len(gt)
	for i in range(total):
		if gt[i] == predict[i]:
			correct_count += 1
			
	accuracy = float(correct_count/total)
	return accuracy
#----------------------------

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
   
   
    #------------------------------
    YY_ = sess.run(YY, feed_dict={X: batch_X, Y_: batch_Y, step: i})
    #print("batch_X",batch_X)
    #print("batch_Y",batch_Y)
    print("YY shape",YY_.shape) 
    print("YY",YY_)
    labels = get_label(batch_Y)
    print("labels",labels)
    activates = get_activates(YY_, 100)
    #print("activates",activates)
    #compare(activates)
    
    learned_patterns = get_common(labels,activates)
    
    #test
    
    batch_X, batch_Y = mnist.train.next_batch(100)
    test_labels = get_label(batch_Y)
    print("test labels",test_labels)
    
    YY_predict = sess.run(YY, feed_dict={X: batch_X, Y_: batch_Y, step: i})
    activates = get_activates(YY_predict, 10)   #activates [[4,10,...500],[]]
    print("activates",activates[:10])
    print()
    predicted = predict(learned_patterns,activates)
    print("predicted",predicted)
    
    acc = get_score(test_labels,predicted)
    print("accuracy",acc)
    sys.exit()
    #-------------------------------
    
   
    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b, l = sess.run([accuracy, cross_entropy, I, allweights, allbiases,lr],
                                  feed_dict={X: batch_X, Y_: batch_Y, step: i})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})

datavis.animate(training_step, 10001, train_data_update_freq=10, test_data_update_freq=100)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# layers 4 8 12 200, patches 5x5str1 5x5str2 4x4str2 best 0.989 after 10000 iterations
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 best 0.9892 after 10000 iterations
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 best 0.9908 after 10000 iterations but going downhill from 5000 on
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75 best 0.9922 after 10000 iterations (but above 0.99 after 1400 iterations only)
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9914 at 13700 iterations
# layers 9 16 25 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9918 at 10500 (but 0.99 at 1500 iterations already, 0.9915 at 5800)
# layers 9 16 25 300, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9916 at 5500 iterations (but 0.9903 at 1200 iterations already)
# attempts with 2 fully-connected layers: no better 300 and 100 neurons, dropout 0.75 and 0.5, 6x6 5x5 4x4 patches no better
#*layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 dropout=0.75 best 0.9928 after 12800 iterations (but consistently above 0.99 after 1300 iterations only, 0.9916 at 2300 iterations, 0.9921 at 5600, 0.9925 at 20000)
# layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 no dropout best 0.9906 after 3100 iterations (avove 0.99 from iteration 1400)
