from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv2D, Flatten, Concatenate, Multiply
import tensorflow as tf

def min_distance_loss(w,epsilon=1.0):
        
    r = tf.reduce_sum(w*w, 1)

    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(w, tf.transpose(w)) + tf.transpose(r)
    D = D + tf.linalg.diag(epsilon * tf.ones(D.shape[0]))
    return tf.reduce_sum(tf.where(D<epsilon,1.0,0.0))/tf.cast(w.shape[1],tf.float32)

def TransE(entities,relations,dim=200,bias=1,lamb=1,norm_size=0.0,mdl=0.0):
    
    inp = Input((3,))
    inp_label = Input(())
    
    s,p,o = tf.unstack(inp,axis=-1)
    
    entity_embedding = Embedding(len(entities),dim,name='entity_embedding')
    relation_embedding = Embedding(len(relations),dim,name='relation_embedding')
    
    h,r,t = entity_embedding(s),relation_embedding(p),entity_embedding(o)
    
    score = bias - tf.norm(h+r-t, ord=2, axis=-1)
    
    loss = lamb - inp_label * score
    loss = tf.where(loss>0,loss,0) + \
    norm_size * tf.norm(entity_embedding.weights[0],ord=2)**2 + \
    min_distance_loss(entity_embedding.weights[0]) * mdl
    
    model = Model(inputs=[inp,inp_label],outputs=score)
    model.add_loss(loss)
    model.compile(optimizer='adam',loss=None)
    
    return model

def DistMult(entities,relations,dim=200,norm_size=0.0,mdl=0.0):
    inp = Input((3,))
    inp_label = Input(())
    
    s,p,o = tf.unstack(inp,axis=-1)
    
    entity_embedding = Embedding(len(entities),dim,name='entity_embedding')
    relation_embedding = Embedding(len(relations),dim,name='relation_embedding')
    
    h,r,t = entity_embedding(s),relation_embedding(p),entity_embedding(o)
    
    score = tf.keras.layers.Activation('linear')(tf.reduce_sum(h*r*t,axis=-1))
    
    model = Model(inputs=[inp,inp_label],outputs=score)
    
    loss = lambda true,pred: tf.reduce_sum(tf.math.log(1+tf.math.exp(-true*pred))) + \
    norm_size * tf.norm(entity_embedding.weights[0],ord=2)**2 + \
    min_distance_loss(entity_embedding.weights[0],mdl) * mdl
    
    model.compile(optimizer='adam',loss=loss)
    
    return model

def ComplEx(entities,relations,dim=200,norm_size=0.0,mdl=0.0):
    inp = Input((3,))
    inp_label = Input(())
    
    s,p,o = tf.unstack(inp,axis=-1)
    
    entity_embedding = Embedding(len(entities),dim,name='entity_embedding')
    relation_embedding = Embedding(len(relations),dim,name='relation_embedding')
    
    h,r,t = entity_embedding(s),relation_embedding(p),entity_embedding(o)
    
    h_real,h_img = tf.split(h,2,axis=-1)
    r_real,r_img = tf.split(r,2,axis=-1)
    t_real,t_img = tf.split(t,2,axis=-1)
    
    score = tf.reduce_sum(r_real*h_real*t_real,axis=-1) + \
    tf.reduce_sum(r_real*h_img*t_img,axis=-1) + \
    tf.reduce_sum(r_img*h_real*t_img,axis=-1) - \
    tf.reduce_sum(r_img*h_img*t_real,axis=-1)
        
    model = Model(inputs=[inp,inp_label],outputs=score)
    
    loss = lambda true,pred: tf.reduce_sum(tf.math.log(1+tf.math.exp(-true*pred))) + \
    norm_size * tf.norm(entity_embedding.weights[0],ord=2)**2 + \
    min_distance_loss(entity_embedding.weights[0]) * mdl
    
    model.compile(optimizer='adam',loss=loss)
    
    return model


def ConvE(entities,relations):
    dim = 200
    inp = Input((3,))
    inp_label = Input(())
    
    s,p,o = tf.unstack(inp,axis=-1)
    
    entity_embedding = Embedding(len(entities),dim,name='entity_embedding')
    relation_embedding = Embedding(len(relations),dim,name='relation_embedding')
    
    h,r,t = entity_embedding(s),relation_embedding(p),entity_embedding(o)
    
    h = tf.reshape(h,(-1,20,10,1))
    r = tf.reshape(r,(-1,20,10,1))
    
    x = Concatenate(axis=2)([h,r])
    
    x = Conv2D(16,(5,5),activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16,(3,3),activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    
    x = Dense(dim)(x)
    x = Multiply()([x,t])
    
    x = Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs=[inp,inp_label],outputs=x)
    
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05))
    
    return model

