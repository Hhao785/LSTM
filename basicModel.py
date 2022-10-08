import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def generate():
    x, y = [], []
    for i in range(0, 1000):
        t1, t2 = random.randint(0, 1000), random.randint(0, 1000)
        x.append([t1, t2])
        y.append(t1+t2)
    return np.array(x), np.array(y)

X, Y = generate()

model = Sequential()
model.add(Dense(1, input_dim=2))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

model.fit(X[:800], Y[:800], batch_size=1, epochs=10)

scores = model.evaluate(x=X[70:], y=Y[70:])
print(scores)

test = np.array([[1000, 2000]])
print(model.predict(test))
