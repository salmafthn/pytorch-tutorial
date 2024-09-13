import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

loss_list = []

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784  # ukuran input untuk gambar input 28x28 yang diratakan menjadi vektor 784
hidden_size1 = 500   # ukuran dari hidden layer 1
hidden_size2 = 300 # ukuran dari hidden layer 2
num_classes = 10  # jumalh kelas output
num_epochs = 10  # jumalah epoch untuk pelatihan
batch_size = 64  # jumlah batch yang diproses setiap iterasi epoch
learning_rate = 0.001  # kecepatan pembelajaran untuk optimisasi


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) # ini adalah layer pertama - dari input menuju hidden layer
        self.relu = nn.ReLU() # ini adalah fungsi aktivasi 1
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) #ini adalah layer kedua -hidden layer
        self.sigmoid = nn.Sigmoid() # ini adalah fungsi aktivasi tambahan (menggunakan sigmoid)
        self.fc3 = nn.Linear(hidden_size2, num_classes) #ini adalah layer tambahan -hidden layer
    
    def forward(self, x):
        out = self.fc1(x) # menjalankan layer pertama ke input
        out = self.relu(out) # menjalankan fungsi aktivasi 
        out = self.fc2(out) # menjalankan layer kedua menuju fungsi aktivasi 2
        out = self.sigmoid(out)  # menjalankan fungsi aktivasi Sigmoid
        out = self.fc3(out) # menjalankan layer kedua menuju output
        return out 

model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # loss function untuk cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # Adam optimizer

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device) #mengubah gambar 28x28 menjadi vektor
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) 
        loss = criterion(outputs, labels) # menjalankan loss function untuk menghitung nilai loss
        # Forward pass digunakan untuk menghasilkan output dari model dan menghitung nilai loss dari hasilnya
        
        # Backward and optimize
        optimizer.zero_grad() # atur ulang gradient sebelum melakukan backpropagation
        loss.backward() # menghitung gradient nya
        optimizer.step() # mengupdate parameter model dengan optimisasi
        # backpropagation berasal dari loss yang dihitung, lalu menghitung gradient untuk mengupdate bobot
        
        loss_list.append(loss.item())
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # ini adalah proses pelatihan dimana kita mengukur seberapa baik model bekerja berdasarkan loss

#plot untuk analisis konvergensi
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device) # Sama seperti sebelumnya, kita ratakan gambar
        labels = labels.to(device)
        outputs = model(images) # hasil prediksi model
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # hitung prediksi yang benar

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


# ini adalah grid search untuk mencari hyperparameter terbaik

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# param_grid = {
#     'batch_size': [64, 100],
#     'learning_rate': [0.001, 0.01],
#     'num_epochs': [5, 10]
# }

# train_dataset = torchvision.datasets.MNIST(root='../../data', 
#                                            train=True, 
#                                            transform=transforms.ToTensor(),  
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data', 
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# best_accuracy = 0
# best_params = None
# for params in ParameterGrid(param_grid):
#     batch_size = params['batch_size']
#     learning_rate = params['learning_rate']
#     num_epochs = params['num_epochs']

#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                                batch_size=batch_size, 
#                                                shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                               batch_size=batch_size, 
#                                               shuffle=False)
    
#     class NeuralNet(nn.Module):
#         def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
#             super(NeuralNet, self).__init__()
#             self.fc1 = nn.Linear(input_size, hidden_size1)
#             self.relu = nn.ReLU()
#             self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#             self.sigmoid = nn.Sigmoid()
#             self.fc3 = nn.Linear(hidden_size2, num_classes)
        
#         def forward(self, x):
#             out = self.fc1(x)
#             out = self.relu(out)
#             out = self.fc2(out)
#             out = self.sigmoid(out)
#             out = self.fc3(out)
#             return out 

#     model = NeuralNet(input_size=784, hidden_size1=500, hidden_size2=300, num_classes=10).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     loss_list = []
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.reshape(-1, 28*28).to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)
  
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             loss_list.append(loss.item())
            
#             if (i+1) % 100 == 0:
#                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    

#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.reshape(-1, 28*28).to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         accuracy = 100 * correct / total
#         print(f'Accuracy with batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}: {accuracy:.2f}%')

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_params = params

# print(f'Best accuracy: {best_accuracy:.2f}%')
# print(f'Best parameters: {best_params}')
