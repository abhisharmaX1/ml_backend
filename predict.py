import pickle
import numpy as np

# Load the trained model from the .pkl file
with open('scaler.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the model to make a prediction
X_new = np.array([[31, 30, 29, 31, 30, 31, 30, 29, 31, 30]])
print(model)
y_new = model.predict(X_new)
# print(y_new)
print("Predicted length of next cycle:", y_new[0])
