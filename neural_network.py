import os
import sys
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def initialize_weights(layers, units_per_layer, inputs, ouput):
    network = []
    layer = []

    network.append([])

    if layers == 2:
        for i in range(len(ouput)):
            temp = []
            for j in range(len(inputs)):
                temp.append(np.random.uniform(-0.5, 0.5))
            layer.append(temp)
        
        network.append(layer)
        
        
        return network
    
    for i in range(units_per_layer):
        temp = []
        for j in range(len(inputs)):
            temp.append(np.random.uniform(-0.5, 0.5))
        layer.append(temp)

    network.append(layer)
    
    for i in range(layers - 3):
        layer = []
        for j in range(units_per_layer):
            temp = []
            for k in range(units_per_layer):
                temp.append(np.random.uniform(-0.5, 0.5))
            layer.append(temp)
        network.append(layer)

    layer = []

    for i in range(len(ouput)):
        temp = []
        for j in range(units_per_layer):
            temp.append(np.random.uniform(-0.5, 0.5))
        layer.append(temp)
    network.append(layer)
        
    return network



def initialize_bias(layers, units_per_layer, inputs, ouput):
    network = []

    network.append([])

    if layers == 2:
        temp = []
        for i in range(len(ouput)):
            temp.append(np.random.uniform(-0.5, 0.5))
        network.append(temp)
            
        return network

    for _ in range(layers - 2):
        temp = []
        for i in range(units_per_layer):
            temp.append(np.random.uniform(-0.5, 0.5))
        network.append(temp)
    
    temp = []
    for i in range(len(ouput)):
        temp.append(np.random.uniform(-0.5, 0.5))

    network.append(temp)

    return network

def predict(weights, bias, x_n):
    outputs = []
    for i in range(1, len(weights)):
        for j in range(len(weights[i])):
            wtx = bias[i][j] + np.dot(weights[i][j], x_n)
            outputs.append(sigmoid(wtx))
        x_n = outputs
        outputs = []
    
    return x_n


def error(weights, bias, inputs, targets):
    z_xn = []

    for i in range(len(inputs)):
        z_xn.append(predict(weights, bias, inputs[i]))

    return (np.square(np.subtract(targets, z_xn))).mean()
        
def final_prediction(weights, bias, testing_data, testing_labels):
    pos = 0
    for i in range(len(testing_data)):
        prediction = predict(weights, bias, testing_data[i])
        acc = 0
        curr = -1
        count = 0
        if np.argmax(prediction) == np.argmax(testing_labels[i]):
            for j in range(len(prediction)-1, 0, -1):
                if curr == -1:
                    curr = prediction[j]
                    count = 1
                elif curr == prediction[j]:
                    count = count + 1
                else:
                    break
                acc = 1 / float(count)
        else: 
            acc = 0
        pos = pos + acc
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f'% (i + 1, np.argmax(prediction) + 1, np.argmax(testing_labels[i]) + 1, acc))
    
    print('\nclassification accuracy=%6.4f\n' % (float(pos) / len(testing_data)))



def neural_network():
    training_path = sys.argv[1]
    testing_path =  sys.argv[2]
    layers = int(sys.argv[3])
    units_per_layer = int(sys.argv[4])
    rounds = int(sys.argv[5])
    learning_rate = 1

    if not os.path.exists(training_path):
        print('Invalid File Training File Path')
        exit()
    training_file = open(training_path)
    training_lines = training_file.readlines()
    if not os.path.exists(testing_path):
        print('Invalid File Testing File Path')
        exit()
    testing_file = open(testing_path)
    testing_lines = testing_file.readlines()

    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []

    classes = {}

    for line in training_lines:
        line = line.split()
        build = []
        for i in range(len(line) - 1):
            build.append(float(line[i]))
        training_data.append(build)
        if float(line[-1]) not in classes:
            classes[float(line[-1])] = []
        training_labels.append([float(line[-1])])
    
    index = 0

    for val in classes:
        build = []
        for i in range(len(classes)):
            if i == int(val) - 1:
                build.append(1)
            else:
                build.append(0)
        classes[val] = build

    for line in testing_lines:
        line = line.split()
        build = []
        for i in range(len(line) - 1):
            build.append(float(line[i]))
        testing_data.append(build)
        testing_labels.append([float(line[-1])])

    for i in range(len(testing_labels)):
        testing_labels[i] = classes[testing_labels[i][0]]
    
    for i in range(len(training_labels)):
        training_labels[i] = classes[training_labels[i][0]]

    max_val = float('-inf')

    for i in range(len(training_data)):
        curr = max(training_data[i])
        if(curr > max_val):
            max_val = curr
    
    for i in range(len(testing_data)):
        curr = max(testing_data[i])
        if(curr > max_val):
            max_val = curr

    for i in range(len(training_data)):
        for j in range(len(training_data[i])):
            training_data[i][j] = training_data[i][j] / max_val

    for i in range(len(testing_data)):
        for j in range(len(testing_data[i])):
            testing_data[i][j] = testing_data[i][j] / max_val

    weights = initialize_weights(layers, units_per_layer, training_data[0], training_labels[0])

    bias = initialize_bias(layers, units_per_layer, training_data[0], training_labels[0])

    last_error = error(weights, bias, training_data, training_labels)

    for _ in range(rounds):
        for n in range(len(training_data)):
            z = list(range(layers))
            z[0] = list(range(len(training_data[n])))

            for i in range(len(training_data[0])):
                z[0][i] = training_data[n][i]

            a = list(range(layers))

            a[0] = list(range(len(training_data[n])))

            for i in range(len(training_data[0])):
                a[0][i] = training_data[0][i]

            for l in range(1, layers):
                a[l] = list(range(len(weights[l])))
                z[l] = list(range(len(weights[l])))
                for i in range(len(weights[l])):
                    a[l][i] = bias[l][i] + np.dot(weights[l][i], z[l-1])
                    z[l][i] = sigmoid(a[l][i])
            
            sigma = list(range(layers))
            sigma[layers-1] = list(range(len(classes)))

            for i in range(len(z[layers-1])):
                sigma[layers - 1][i] = (z[layers-1][i] - training_labels[n][i]) * z[layers-1][i] * (1-z[layers-1][i])
            
            for l in range(layers-2, 0, -1):
                sigma[l] = list(range(len(z[l])))
                for i in range(len(z[l])):
                    summation = 0
                    for k in range(len(z[l+1])):
                        summation = summation + sigma[l+1][k]*weights[l+1][k][i]
                    sigma[l][i] = summation * z[l][i] * (1-z[l][i])
            
            for l in range(1, layers):
                for i in range(len(z[l])):
                    bias[l][i] = bias[l][i] - learning_rate * sigma[l][i]
                    for j in range(len(z[l-1])):
                        weights[l][i][j] = weights[l][i][j] - learning_rate * sigma[l][i] * z[l-1][j]  
        learning_rate = learning_rate * 0.98

    final_prediction(weights, bias, testing_data, testing_labels)
    

neural_network()