
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#config InlineBackend.figure_formats = {'png', 'retina'}
import os
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#5x5枚の画像を表示する
plt.figure(figsize=(10,10))
for i in range(25):
    rand_num=np.random.randint(0,50000)
    cifar_img=plt.subplot(5,5,i+1)
    plt.imshow(x_train[rand_num])
    #x軸の目盛りを消す
    plt.tick_params(labelbottom='off')
    #y軸の目盛りを消す
    plt.tick_params(labelleft='off')
    #正解ラベルを表示
    plt.title(y_train[rand_num])


plt.show()

"""
# データ型の変換＆正規化
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# one-hot変換
num_classes = 10 
y_train = to_categorical(y_train, num_classes = num_classes)
y_test = to_categorical(y_test, num_classes = num_classes)

model = Sequential()

model.add(Conv2D(
    32, # フィルター数（出力される特徴マップのチャネル）
    kernel_size = 3, # フィルターサイズ
    padding = "same", # 入出力サイズが同じ
    activation = "relu", # 活性化関数
    input_shape = (32, 32, 3) # 入力サイズ
))
model.add(Conv2D(
    32,
    kernel_size = 3,
    activation = "relu"
))
# 各特徴マップのチャネルは変わらず、サイズが1/2
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(
    64,
    kernel_size = 3,
    padding = "same", 
    activation = "relu"
))
model.add(Conv2D(
    64,
    kernel_size = 3,
    activation = "relu"
))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

# 全結合層（fully-connected layers）につなげるため、
# マトリックスデータ（多次元配列）である特徴マップを多次元ベクトルに変換（平坦化）
model.add(Flatten())
# サイズ512のベクトル（512次元ベクトル）を出力
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
# クラス数のベクトルを出力
model.add(Dense(num_classes))
model.add(Activation("softmax"))

optimizer = Adam(lr = 0.001)
model.compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log/")

# モデルの学習
hist = model.fit(
    X_train,
    y_train,
    verbose = 1,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = [early_stopping, reduce_lr, logging]
)

model_dir = './model/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

# optimizerのない軽量モデルを保存（学習や評価は不可だが、予測は可能）
model.save(model_dir + 'model-opt.hdf5', include_optimizer = False)

# ベストの重みのみ保存
model.save_weights(weights_dir + 'model_weight.hdf5')
"""
