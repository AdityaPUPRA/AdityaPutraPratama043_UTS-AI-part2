# Multi Neuron Batch Input

#Inisialisasi nummpy
import numpy as np

# Inisialisasi Variabel Input
# 6 batcth Inputs setiap batch berisi 10 
inputs =[   
            # Inputs 1
            [4.2,-5.0,4.0,-1.22,3.5,6.5,4.9,2.13,5.8,-3.55],
            # Inputs 2
            [2.5,3.3,-2.25,4.2,3.0,5.25,4.1,3.2,7.0,-1.11],
            # Inputs 3
            [1.1,-3.05,3.2,9.0,1.25,3.75,-1.23,0.15,-4.23,1.15],
            # Inputs 4
            [5.25,-5.25,1.15,-1.25,8.9,10.3,1.88,3.4,-0.17,0.21],
            # Inputs 5
            [3.0,0.49,0.33,0.34,-0.12,0.46,0.93,-0.28,3.44,1.92],
            # Inputs 6
            [2.34,-1.14,2.27,8.09,6.61,7.24,1.91,-2.22,4.78,0.75]
        ]

# Inisialisasi Variabel Weights 1
weights_1 =[
            # Neuron 1
            [1.25,-4.1,-2.3,-9.19,4.45,1.67,2.39,-7.14,0.99,0.3],
            # Neuron 2
            [-0.19,9.81,9.99,1.34,2.55,-2.82,1.56,-0.85,2.09,7.04],
            # Neuron 3
            [1.12,7.56,-2.25,5.67,1.04,9.15,3.14,9.91,-1.55,2.45],
            # Neuron 4
            [-1.2,4.44,5.35,-1.17,3.55,9.02,-1.45,2.49,-1.11,9.02],
            # Neuron 5
            [-6.78,2.18,-4.44,8.06,-1.01,2.09,1.23,-4.04,7.34,1.50]
        ] 

# Inisialisasi bias 1
# Jumlah bias pada layer 1 berisi 5
biases_1 = [1.54,2.25,2.15,5.91,1.23]

# Inisialisasi Variabel Weights 2
# Jumlah neuron sesuai dengan jumlah bias pada layer ke 2, yaitu 3
# Di setiap neuron sesuai dengan jumlah bias pada layer 1, yaitu 5
Weights_2 =[
            # Neuron 1
            [2.05,3.45,2.51,9.12,2.46],
            # Neuron 2
            [3.5,6.8,2.59,6.12,3.66],
            # Neuron3 
            [4.05,2.34,0.39,1.45,3.24]
        ] 

# Inisialisasi bias 2
# Jumlah bias pada layer 2 berisi 3
biases_2 = [0.5,3.99,4.25]

# Perhitungan output layer 1 
layer_outputs_1 = np.dot(inputs, np.array(weights_1).T) + biases_1

# Perhitungan output layer 1 
layer_outputs_2 = np.dot(layer_outputs_1,np.array(Weights_2).T)+ biases_2

# Print layer_output 2
print(layer_outputs_2)