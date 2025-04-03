#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topologySize = topology.size();
	this->topology = topology;

	for (int i = 0; i < topology.size(); i++) {
		Layer *l = new Layer(topology.at(i));
		this->layers.push_back(l);
	}

	for (int i = 0; i < topology.size() - 1; i++) {
		Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);
		this->weightMatrices.push_back(m);
	}

}

void NeuralNetwork::feedForward() {
	for (int i = 0; i < (this->layers.size() - 1); i++) {
		Matrix *a = this->getNeuronMatrix(i);

		if (i != 0) {
			a = this->getActivatedNeuronMatrix(i);
		}	

		Matrix *b = this->getWeightMatrix(i);
		Matrix *c = *a * *b;

		for (int k = 0; k < c->getNumCols(); k++) {
			this->setNeuronValue(i + 1, k, c->getVal(0, k));
		}
	} 
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i));
	}
}

void NeuralNetwork::printToConsole() {
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "LAYER: " << i << endl;
		if (i == 0) {
			Matrix *m = this->layers.at(i)->matrixifyVals();
			m->printToConsole();
		}
		else {
			Matrix *m = this->layers.at(i)->matrixifyActivatedVals();
			m->printToConsole();
		}


	}
}
