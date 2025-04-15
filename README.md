# NeuralNetwork

This project implements a feedforward artificial neural network from scratch in C++. It includes training, prediction, saving, and loading models with support for backpropagation using gradient descent.

## Features

- Customizable layer topology
- Manual matrix and layer management
- Save and load model weights/biases to/from file
- Forward and backward propagation
- Console output utilities for debugging

## Getting Started

### Prerequisites

- C++ Compiler (g++ or clang)
- CMake

### Build Instructions

```bash
cmake .
make
./nn_from_scratch
```

## Class: `NeuralNetwork`

### Constructor Overloads

- `NeuralNetwork(vector<int> topology, double learningRate)`  
  Initializes a new neural network with the given layer topology and learning rate.

- `NeuralNetwork(const std::string& path)`  
  Loads a neural network from a saved model file.

### Public Methods

- `void setCurrentInput(vector<double> input)`  
  Sets the current input for the network and assigns it to the first layer.

- `void backPropogate()`  
  Executes backpropagation to adjust weights and biases based on error.

- `void saveModel(const std::string& path)`  
  Saves the model's topology, weights, biases, and learning rate to the specified file.

- `void setWeightMatrix(int index, Matrix* weightMatrix)`  
  Updates the weight matrix at the given layer index.

- `void setBiasMatrix(int index, Matrix* biasMatrix)`  
  Updates the bias matrix at the given layer index.

- `void printInputToConsole()`  
  Prints the input layer’s values to the console.

- `void printOutputToConsole()`  
  Prints the output layer’s values to the console.

- `void printTargetToConsole()`  
  Prints the target values used in backpropagation.

- `void printToConsole()`  
  Prints all layer values, weights, and biases of the entire network.

### Internal Mechanics

- `Matrix* predict(vector<double> input)`  
  Used internally (primarily in `loadModel`) to perform forward propagation on input.

- `void feedForward()`  
  Propagates input forward through all layers.

- `void setErrors()`  
  Computes and stores error values for output neurons using MSE.

## Model File Format

- Layer topology: comma-separated integers followed by `;`
- Weight matrices: each matrix serialized row-by-row, comma-separated, then `;`
- Bias matrices: each vector serialized, comma-separated, then `;`
- Final learning rate appended as the last semicolon-separated value

## Usage Example

```cpp
#include <iostream>
#include <vector>
#include <NeuralNetwork.hpp>

int main() {
    // Define the network topology: 2 input neurons, 3 hidden neurons, 1 output neuron
    std::vector<int> topology = {2, 3, 1};
    
    // Create the neural network with a learning rate of 0.1
    NeuralNetwork net(topology, 0.1);

    // Define input values
    std::vector<double> input = {0.5, 0.8};
    
    // Set the input to the network and perform forward propagation
    net.setCurrentInput(input);
    net.feedForward();

    // Set target values (for training purposes)
    std::vector<double> target = {0.1};  // Example target output
    net.setTarget(target);

    // Perform backpropagation to adjust weights and biases
    net.backPropogate();

    // Save the trained model to a file
    net.saveModel("model.nn");

    std::cout << "Model saved successfully!" << std::endl;

    return 0;
}
```

## References 

[devlogs-Playlist](https://www.youtube.com/watch?v=PQo78WNGiow&list=PL2-7U6BzddIYBOl98DDsmpXiTcj1ojgJG)
