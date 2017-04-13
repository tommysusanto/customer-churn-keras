import pandas
from keras.models import Sequential
from keras.layers import Dense
import numpy

dataframe = pandas.read_csv("train_data.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:19]
Y = dataset[:,20]

model = Sequential()
model.add(Dense(5380, input_dim=19, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=5380)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
