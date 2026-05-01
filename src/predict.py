import pickle
import numpy as np

model = pickle.load(open("models/model.pkl", "rb"))

data = np.array([[5000, 150]])  # sample input

pred = model.predict(data)

print("Prediction:", pred)