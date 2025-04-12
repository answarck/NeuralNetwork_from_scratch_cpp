#include <iostream>
#include <vector>
#include "../include/Neuron.hpp" 
#include "../include/Matrix.hpp" 
#include "../include/NeuralNetwork.hpp" 
using namespace std;


int main(int argc, char** argv) {
	vector<int> topology;
	topology.push_back(5);
	topology.push_back(128);
	topology.push_back(256);
	topology.push_back(10);

	vector<double> input;
	input.push_back(1);
	input.push_back(2);
	input.push_back(3);
	input.push_back(4);
	input.push_back(5);
	vector<double> output;
	output.push_back(10);
	output.push_back(4);
	output.push_back(3);
	output.push_back(2);
	output.push_back(1);
	output.push_back(0);
	output.push_back(9);
	output.push_back(8);
	output.push_back(7);
	output.push_back(6);
	NeuralNetwork *nn = new NeuralNetwork(topology, 0.01);
	nn->setCurrentInput(input);
	nn->setCurrentTarget(output);
	for (int i = 0; i < 600; i++) {
		nn->feedForward();
		nn->backPropogate();
		nn->printOutputToConsole();
	}
	delete nn;

	return 0;
}
