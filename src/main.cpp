#include <iostream>
#include <vector>
#include "../include/Neuron.hpp" 
#include "../include/Matrix.hpp" 
#include "../include/NeuralNetwork.hpp" 
using namespace std;

int main(int argc, char** argv) {
	NeuralNetwork *nn = new NeuralNetwork("./model.model");
	
	vector<double> input;
	input.push_back(3.0);
	input.push_back(3.0);
	input.push_back(3.0);
	input.push_back(1.0);
	input.push_back(2.0);

	nn->predict(input)->transpose()->printToConsole();
	delete nn;

	return 0;
}
