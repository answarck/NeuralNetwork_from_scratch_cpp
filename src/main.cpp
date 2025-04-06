#include <iostream>
#include <vector>
#include "../include/Neuron.hpp" 
#include "../include/Matrix.hpp" 
#include "../include/NeuralNetwork.hpp" 
using namespace std;

int main(int argc, char** argv) {
	vector<int> topology;
	topology.push_back(5);
	topology.push_back(10);
	topology.push_back(5);

	vector<double> input;
	input.push_back(3.0);
	input.push_back(3.0);
	input.push_back(3.0);
	input.push_back(1.0);
	input.push_back(2.0);

	NeuralNetwork *nn = new NeuralNetwork(topology, 0.01);
	nn->setCurrentInput(input);
	nn->setCurrentTarget(input);

	for (int i = 0; i < 30; i++) {
		cout << "__________________________" << endl;
		cout << "EPOCH: " << i << endl;
		nn->feedForward();
		nn->backPropogate();
		nn->printOutputToConsole();
		nn->printTargetToConsole();
		cout << "Total Error: " << nn->getError() << endl;
		cout << "__________________________" << endl;
	}

	delete nn;

	return 0;
}
