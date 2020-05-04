from gensim.models import Word2Vec
import multiprocessing as mul
import numpy as np
import time
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Flatten,Input,Embedding
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
# vector of line 
def Sourcefile_CombineVector(word_sent,tf_idf,model):
    vector_line= model.wv[word_sent]*tf_idf['tfidf'][word_sent] 
    return vector_line
def mutliprocessing_sourcefile(corpus,model,tf_idf):
    #compute line max
    line_max=0
    for ptu in corpus:
      if(len(ptu)>line_max):
        line_max=len(ptu)
    matrixs=Sourcefile_Word_embedding(corpus,tf_idf,line_max,model)
    return matrixs
def Sourcefile_Word_embedding(group,tf_idf,line_max,model):
    matrix=[]
    for sourcefile in group:
        vecto=np.zeros(shape=(line_max,300))
        for i in range(0,len(sourcefile)):
            vecto[i]= Sourcefile_CombineVector(sourcefile[i],tf_idf,model)
        matrix.append(vecto)
    return matrix

#extract vector with CNN
def feature_detect (matrix):
    vectors=[]
    for source in matrix:
        input = np.expand_dims(source, axis=0)
        vector_feature = np.zeros((3, 100))
        num = 0
        for i in range(2, 5):
            model = Sequential()
            model.add(Conv1D(activation='relu', filters=100, kernel_size=i, input_shape=source.shape))
            model.add(MaxPooling1D(pool_size=source.shape[0] - i + 1))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            vector = np.reshape(model.predict(input), (1, 100))
            vector_feature[num] = vector
            num += 1
        vectors.append(vector_feature.flatten())
    return vectors
def sourcefile_extractvector(srs,tf_idf):
    model=Word2Vec.load("new_w2v_model.model")
    res=mutliprocessing_sourcefile(srs,model,tf_idf)
    vectors=feature_detect(res)
    return vectors