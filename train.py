import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
import utils
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data = pd.read_csv('data.csv', sep=';')
logging.info("Data loaded.")

data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')

consumption = pd.DataFrame({'consumption': data['consumption']})
data['Seconds'] = data.index.map(pd.Timestamp.timestamp)

day = 60*60*24
year = 365.2425*day

data['Year sin'] = np.sin(data['Seconds'] * (2 * np.pi / year))
data['Year cos'] = np.cos(data['Seconds'] * (2 * np.pi / year))
data['consumption'] = StandardScaler().fit_transform(data['consumption'].values.reshape(-1, 1)).flatten()
logging.info("Data processed.")

data = data.drop(["Date", "Seconds"], axis=1)

X, y = utils.df_to_X_y2(data)
logging.info("Data scaled.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data splitted.")

model = Sequential()
model.add(InputLayer((6, 3)))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(Conv1D(64, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))
logging.info("Model created.")
logging.info(model.summary())

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
logging.info("Model compiled.")

logging.info("Model training...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300)
logging.info("Model trained.")

saved_model_path = "./models"
tf.saved_model.save(model, saved_model_path)
logging.info("Model saved.")

loss = model.evaluate(X_test, y_test)

# Save results into results.txt
with open('results.txt', 'w') as f:
    f.write("\nMeanSquaredError = " + str(loss[0]))