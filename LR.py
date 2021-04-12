import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers

#%matplotlib inline

data = pd.read_csv("G:\df.csv", ) # 获取数据
# (2)获取第1，2列
col_12 = data[["rating","friendCount","reviewCount","firstCount","usefulCount","coolCount","funnyCount","complimentCount","tipCount","fanCount","flagged"]]  #获取多列，要用二维数据
data_12 = np.array(col_12)
print(data_12)
x = col_12.iloc[:, :-1]
y = col_12.iloc[:, -1].replace('N', 0).replace('Y',1)# N -0是正常评论  Y -1是虚假评论

data.head()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape = (10, ), activation = 'relu'))
model.add(tf.keras.layers.Dense(4, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['acc'] # 设置显示的参数
)


history = model.fit(x, y, epochs = 10) # 训练1000次

print(history.history.keys())

plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))
plt.show()
