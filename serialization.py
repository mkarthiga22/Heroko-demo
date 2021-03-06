import pickle

# Serializing the model
from Linear_Regression_Model import linreg

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(linreg, f)

# De-Serializing the model
with open('trained_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

# Check the pickle file by inputing the variables
model = pickle.load(open('trained_model.pkl', 'rb'))
print(model.predict([[55, 18, 0, 1, 1, 1]]))