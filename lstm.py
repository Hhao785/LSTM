from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from pandas_datareader import DataReader
from sklearn.preprocessing import MinMaxScaler

# read data
print("Loading data...")
df = DataReader("AAPL", data_source="yahoo", start="2020-01-01", end="2022-09-01")
df = df.sort_index(ascending=True, axis=0)
# Split the data into training and testing sets
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


# Build the LSTM model
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# lstmModel = build_model(trainX, output_size=1, neurons=20)
# lstmModel.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
