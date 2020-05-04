import pandas as pd
import re
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import precision_recall_fscore_support 
tf.compat.v1.disable_eager_execution()

def arrangeLabel(y_pred):
    x=[]
    for y_predict in y_pred:
      if(y_predict[0]>y_predict[1]):
        x.append(0)
      else:
        x.append(1)
    return x

def returnRiFi():
    fi = pd.read_csv(r'/content/drive/My Drive/Wiki_word2vec/NewData/InputForEnhanceCNN_label1.csv')
    r1 = fi['Ri'].values
    f1 = fi['Fi'].values
    r1 = r1.tolist()
    f1 = f1.tolist()
    fo = pd.read_csv(r'/content/drive/My Drive/Wiki_word2vec/NewData/InputForEnhanceCNN_label0.csv')
    r2 = fo['Ri'].values
    f2 = fo['Fi'].values
    for i in r2:
        r1.append(i)
    x= np.asarray(r1)

    for i in f2:
        f1.append(i)
    y= np.asarray(f1)
    return x,y

def arrangeLabel(y_pred):
    x=[]
    for y_predict in y_pred:
      if(y_predict[0]>y_predict[1]):
        x.append(0)
      else:
        x.append(1)
    return x



def returnLabel(count):
    fi = pd.read_csv(r'/content/drive/My Drive/Wiki_word2vec/NewData/InputForEnhanceCNN_label1.csv')
    x = fi['label']
    x= len(x)
    arr=[]
    for i in range(x):
        arr.append([0.,1.])
    for i in range(count-x):
        arr.append([1.,0.])
    arr= np.asarray(arr)
    return arr


def returnInput():
    vector =[]
    file = pd.read_csv(r'/content/drive/My Drive/Wiki_word2vec/NewData/InputForEnhanceCNN_label1.csv')
    x = file['input_vector'].values
    count=len(x)
    for i in x:
        mn = to_matrix(i, 2, 300)
        vector.append(mn)
    file1 = pd.read_csv(r'/content/drive/My Drive/Wiki_word2vec/NewData/InputForEnhanceCNN_label0.csv')
    y = file1['input_vector'].values
    count+= len(y)
    for i in y:
        mn = to_matrix(i, 2, 300)
        vector.append(mn)
    return vector


def to_matrix(input, shape1, shape2):
    matrix = np.zeros(shape=(shape1, shape2))
    mtrx = re.sub("\[", ' ', input)
    mtrx = re.sub("]", ' ', mtrx)
    mtrx = mtrx.split()
    count_col = 0
    count_row = 0
    for i in mtrx:
        j = float(i)
        if (count_col < shape2):
            matrix[count_row][count_col] = j
        if (count_col == shape2) and (count_row < shape1):
            count_col = 0
            count_row += 1
            matrix[count_row][count_col] = j
        count_col += 1
    return matrix.tolist()

# layer convolution 2d
def con2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

# layer maxpooling 2d
def maxpooling2d (x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# mang CNN voi 1 conv2d,1 maxpooling 2d va 2 hidden layer 
def conv_net(x,weights,biases,drop_out):
    conv1=con2d(x,weights['wc1'],biases['bc1'])
    conv1=maxpooling2d(conv1,k=2)
    fc1=tf.reshape(conv1,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1= tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1= tf.nn.tanh(fc1)
    fc1=tf.nn.dropout(fc1,drop_out)
    fc2=tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    fc2=tf.nn.tanh(fc2)
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

# loss function bo sung them he so ri,fi
def enhance_loss(labels,logits,weights,rf):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)-tf.matmul(rf,weights['w_rf']))
 
    
if __name__ == "__main__":
    
    # Khoi tao cac gia tri 
    learning_rate= 0.001
    epochs=10
    batch_size= 100
    n_classes=2
    drop_out=0.75
    
    # Store layers weight & bias
    weights = {
        # 2x2 conv, 1 input, 100 outputs
        'wc1': tf.Variable(tf.random.normal([2, 2, 1, 100])),
        # fully connected, 1*150*100 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random.normal([1*150*100, 1024])),
        'wd2': tf.Variable(tf.random.normal([1024,256])),
        # 1024 inputs, 2 outputs (class prediction)
        'out': tf.Variable(tf.random.normal([256, n_classes])),
        'w_rf':tf.Variable(tf.random.truncated_normal([2,1],dtype=tf.float32))
    }

    biases = {
        'bc1': tf.Variable(tf.random.normal([100])),
        'bd1': tf.Variable(tf.random.normal([1024])),
        'bd2': tf.Variable(tf.random.normal([256])),
        'out': tf.Variable(tf.random.normal([n_classes])),
    }
    
    
    # load data: input,label, ri,fi
    vectors=returnInput()
    vector=[]
    for i in vectors:
        vector.append(np.reshape(i,(2,300)))
    vector=np.reshape(vector,(len(vector),2,300,1)) # vecto input
    label=returnLabel(len(vector))                  # label

    r,f=returnRiFi()
    r=np.reshape(r,(r.shape[0],1))
    f=np.reshape(f,(f.shape[0],1))
    rf=np.concatenate([r,f],axis=1)                  # ri,fi
    vector_train,vector_vt,label_train,label_vt,rf_train,rf_vt=train_test_split(vector,label,rf,test_size=0.4)
    vector_val,vector_test,label_val,label_test,rf_val,rf_test=train_test_split(vector_vt,label_vt,rf_vt,test_size=0.5)



    # input model
    x= tf.compat.v1.placeholder(tf.float32,[None,2,300,1])
    y= tf.compat.v1.placeholder(tf.float32,[None,n_classes])
    z= tf.compat.v1.placeholder(tf.float32,[None,2])
    keep_prob=tf.compat.v1.placeholder(tf.float32)
    
    #su dung mang da tao tinh predict
    predict=conv_net(x,weights,biases,drop_out)
    predict=tf.nn.softmax(predict)

    # Define loss and optimizer
    cost=enhance_loss(y,predict,weights,z)
    # cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    saver=tf.compat.v1.train.Saver()

    # Initializing the variables
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
      sess.run(init)
      for epoch in range(1,epochs+1,1):
          # chia batch size 
          for batch_id in range(len(vector_train)//batch_size):
              batch_input= vector_train[batch_id*batch_size:(batch_id+1)*batch_size]
              batch_label= label_train[batch_id*batch_size:(batch_id+1)*batch_size]
              rf=rf_train[batch_id*batch_size:(batch_id+1)*batch_size][:]
              batch_input= np.asarray(batch_input)
              batch_label=np.asarray(batch_label)
              rf=np.asarray(rf)
              #training cung voi du lieu train
              sess.run(optimizer,feed_dict={x:batch_input,y:batch_label,z:rf,keep_prob:drop_out})
          #tinh loss_train, acc_train sau moi epoch
          loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_input,y:batch_label,z:rf,keep_prob:1.})
          #tinh loss_val, acc_val sau moi epoch
          loss_val,acc_val=sess.run([cost,accuracy],feed_dict={x:vector_val,y:label_val,z:rf_val,keep_prob:1.})
          print("Epoch",epoch,":Loss= ","{:.6f}".format(loss), ", Accuracy= ","{:.5f}".format(acc),"|| Loss_val: ","{:.6f}".format(loss_val),", Accuracy_val= ","{:.5f}".format(acc_val))
      y_pred=sess.run(predict,feed_dict={x:vector_test,y:label_test,z:rf_test,keep_prob:1.})
      loss_test,acc_test=sess.run([cost,accuracy],feed_dict={x:vector_test,y:label_test,z:rf_test,keep_prob:1.})
      print("Test Accuracy: ",acc_test,"Test Loss",loss_test)
      
      
          
      
