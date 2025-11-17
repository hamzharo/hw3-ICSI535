import pickle
import CNN

with open("weights_LeNet5_SGD.pkl", "rb") as f:
    params = pickle.load(f)

model = CNN.LeNet5()
model.set_params(params)

print("Type of conv1.W:", type(model.conv1.W))
print("conv1.W:", model.conv1.W)
