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

# valid = np.array([207.122,218.58,217.216,35.5])
valid = np.array([206.996,218.912,217.747,
                207.719,219.893,218.739,
                207.834,220.24,219.159]).reshape(-1,1)

print(valid.shape)
# scaler_train = MinMaxScaler(feature_range=(0, 1))
# valid = scaler_train.fit_transform(valid)
valid = valid.reshape(-1,3)
time_stamp=1
valid_da = pd.DataFrame(data)
y = valid_da.iloc[:,-1]
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

# model.fit(x_train, y_train, epochs=1000, batch_size=128, verbose=1,callbacks=[modelcheckpoint_best])
model.load_weights('best.hdf5')
print(valid.shape)
valid = np.array(valid).reshape(-1,1,3)
gen_datas = model.predict(valid)
# predict = gen_datas*(dataY_6_train.max(axis=0)-dataY_6_train.min(axis=0))+dataY_6_train.min(axis=0)
# real = dataX_6_test*(dataY_6_test.max(axis=0)-dataY_6_test.min(axis=0))+dataY_6_test.min(axis=0)
# real = y_valid*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0)
predict = gen_datas*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0)
print('predict',predict)

# import matplotlib.pyplot as plt
# plt.plot(real,color='b')
# plt.plot(predict,color='r')
# plt.xlabel('epoch')
# plt.ylabel('value')
# plt.legend(['real','predict'])
# plt.show()