import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import *

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import Sequential
from keras.layers import *

# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def build_model():
    inp = Input(shape=(max_len, feature_size))
    x = BatchNormalization()(inp)
    x = Bidirectional(CuDNNGRU(256, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    
    atten_1 = Attention(10)(x)
    atten_2 = Attention(10)(y)
    
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    concat = concatenate([atten_1, atten_2, avg_pool, max_pool])
    concat = Dense(64, activation="relu")(concat)
    concat = Dropout(0.5)(concat)
    output = Dense(1, activation="sigmoid")(concat)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

def predict(X, y, X_test):
	kf = KFold(n_splits=10, shuffle=True, random_state=42069)
	preds = []
	fold = 0
	aucs = 0
	for train_idx, val_idx in kf.split(X):
	    x_train_f = X[train_idx]
	    y_train_f = y[train_idx]
	    x_val_f = X[val_idx]
	    y_val_f = y[val_idx]
	    model = build_model()
	    model.fit(x_train_f, y_train_f,
	              batch_size=256,
	              epochs=16,
	              verbose = 0,
	              validation_data=(x_val_f, y_val_f))
	    # Get accuracy of model on validation data. It's not AUC but it's something at least!
	    preds_val = model.predict([x_val_f], batch_size=512)
	    preds.append(model.predict(X_test))
	    fold+=1
	    fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)
	    aucs += auc(fpr,tpr)
	    print('Fold {}, AUC = {}'.format(fold,auc(fpr, tpr)))
	print("Cross Validation AUC = {}".format(aucs/10))
	preds = np.asarray(preds)[...,0]
	preds = np.mean(preds, axis=0)
	sub_df = pd.DataFrame({'vid_id':test_df['vid_id'].values,'is_turkey':preds})
	return sub_df

# Load data
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
# train_df.head()
# test_df.head()

# Visualize number of turkey and non-turkey in train dataset
plt.figure(figsize=(12,8))
sns.countplot(train_df['is_turkey'])
plt.show()

# Visualize the length of audio_embedding in train dataset
train_df['length'] = train_df['audio_embedding'].apply(len)
plt.figure(figsize=(12,8))
plt.yscale('log')
sns.countplot('length', hue='is_turkey', data= train_df)
plt.show()

# As we can see the maximum length is 10
max_len = 10
feature_size = 128

X = pad_sequences(train_df['audio_embedding'], maxlen = max_len, padding='post')
X_test = pad_sequences(test_df['audio_embedding'], maxlen = max_len, padding='post')
y = train_df['is_turkey'].values

# First predict
sub_df = predict(X, y, X_test)

# Since the model seems to be doing quite well, we tried to experiment with pseudo-labeling (using test data with high confidences as training data), 
# however there's is currently no improvements.
probs = sub_df.is_turkey.values
n,bins,_ = plt.hist(probs,bins=100)
print(n, bins)
pos_threshold = 0.99
neg_threshold = 0.01
pseudo_index = np.argwhere(np.logical_or(probs > pos_threshold, probs < neg_threshold ))[:,0]

pseudo_x_train = X_test[pseudo_index]
pseudo_y_train = probs[pseudo_index]
pseudo_y_train[pseudo_y_train > 0.5] = 1
pseudo_y_train[pseudo_y_train <= 0.5] = 0

X = np.concatenate([X, pseudo_x_train],axis=0)
y = np.concatenate([y,pseudo_y_train])
print(X.shape, y.shape)

# Final predict
predict_df = predict(X, y, X_test)
predict_df.to_csv('submission.csv', index=False)