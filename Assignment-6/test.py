import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
"""
def test(hidden_nodes):
    training_data, validation_data, test_data=mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    return net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

hidden_nodes=[30]
accuracys=[]
for e in hidden_nodes:
    print("Hidden Nodes : ",e)
    accuracys.append(test(e))
accuracy=[]
for e in accuracys:
    accuracy.append(max(e))
plt.plot(hidden_nodes,accuracy)
plt.xlabel("Hidden Nodes")
plt.ylabel("Accuracy")
plt.title("Hidden nodes vs Accuracy")
plt.show()
"""
x=[[0,0],[0,1],[1,0],[1,1]]
y=[[0],[1],[1],[0]]
arr_x=np.array(x)
arr_y=np.array(y)
training_data =zip(arr_x,arr_y)
test_data =zip(arr_x,arr_y)
net = network.Network([2, 2, 1])
accuracy=net.SGD(training_data, 30, 1, 3.0, test_data=test_data)

