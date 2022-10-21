import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf
data = pd.read_csv('data.csv')
# datasets = datasets_all[0:5216]
# print('len(datasets)',len(datasets))
import numpy as np
train = data.tail(-100)
valid = data.tail(100)
print('len(train)',len(train))
print('len(valid)',len(valid))
time_stamp=1
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_train.fit_transform(train)
        # 制作lstm训练数据集
x_train, y_train = [], []
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data[i - time_stamp:i,0:-1])
    y_train.append(scaled_data[i, -1])
x_train, y_train = np.array(x_train), np.array(y_train)
print('x_train',x_train.shape)
        # 验证集
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler_test.fit_transform(valid)
        # 制作lstm验证数据集
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled_data[i - time_stamp:i,0:-1])
    y_valid.append(scaled_data[i, -1])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)
valid_da = pd.DataFrame(valid)
y = valid_da.iloc[:,-1]

# model = Sequential()
# model.add(LSTM(units=100, return_sequences=True, input_dim=dataX_6_train.shape[-1], input_length=dataX_6_train.shape[1]))
# model.add(LSTM(units=100))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(dataX_6_train, dataY_6_train, epochs=1000, batch_size=128, verbose=1).
from tensorflow.keras import layers
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
def fit_lstm():
    num_heads = 14
    projection_dim = 11
    x_input = Input(shape=(1, 3))
    x_input1 = layers.Conv1D(filters=34, kernel_size=1, padding='same', activation='relu')(x_input)
    x_input1 = layers.MaxPool1D(pool_size=1, padding='valid')(x_input1)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.0799
    )(x_input1, x_input1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, x_input1])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=[34], dropout_rate=0.0001)

    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.0055)(representation)
    # Classify outputs.
    logits = layers.Dense(1)(representation)
    logits = layers.Dropout(0.0055)(logits)
    # Create the Keras model.
    return Model(inputs=x_input, outputs=logits)
model = fit_lstm()
model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=0.00055, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0,
                                                     amsgrad=False),
                  metrics=['mse'])
    # In[24]:

    # model.fit(X_train, y_train, batch_size=128, epochs=700, verbose=1)
checkpoint_path_best = "best.hdf5"
modelcheckpoint_best = keras.callbacks.ModelCheckpoint(checkpoint_path_best,
                                  monitor='loss',
                                  save_best_only=True,
                                  mode='min',verbose=0)
# history1 = model.fit(dataX_6_train, dataY_6_train, batch_size=32, epochs=300, verbose=1,callbacks=[modelcheckpoint_best])

model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1,callbacks=[modelcheckpoint_best])
model.load_weights('best.hdf5')
gen_datas = model.predict(x_valid)
# predict = gen_datas*(dataY_6_train.max(axis=0)-dataY_6_train.min(axis=0))+dataY_6_train.min(axis=0)
# real = dataX_6_test*(dataY_6_test.max(axis=0)-dataY_6_test.min(axis=0))+dataY_6_test.min(axis=0)
real = y_valid*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0)
predict = gen_datas*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0)
print('gen_datas',gen_datas.shape)
import matplotlib.pyplot as plt
plt.plot(real,color='b')
plt.plot(predict,color='r')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend(['real','predict'])
plt.show()