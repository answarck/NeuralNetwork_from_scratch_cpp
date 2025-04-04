#include <vector>
#include <cassert>
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

void NeuralNetwork::backPropogate() {
	vector<Matrix *> newWeights;
	// output -> hidden
	int outputLayerIndex      = this->layers.size() - 1;
	Matrix *derivedValuesYToZ = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
	Matrix *gradientYToZ      = new Matrix(1, derivedValuesYToZ->getNumCols(), false);
	for (int i = 0; i < this->errors.size(); i++) {
		double v = derivedValuesYToZ->getVal(0, i);
		double e = this->errors.at(i);
		double g = v * e;
		gradientYToZ->setVal(0, i, g);
	}

	int lastHiddenLayerIndex    = outputLayerIndex - 1;
	Layer *lastHiddenLayer      = this->layers.at(lastHiddenLayerIndex);
	Matrix *weightsOutputHidden = this->weightMatrices.at(lastHiddenLayerIndex);
	Matrix *deltaOutputHidden   = (*gradientYToZ->transpose() * 
				       *lastHiddenLayer->matrixifyActivatedVals())->transpose();
	Matrix *newWeightsOutputToHidden = *weightsOutputHidden - *deltaOutputHidden;

	newWeights.push_back(newWeightsOutputToHidden);

	cout << "Output to Hidden: " << endl;
	newWeightsOutputToHidden->printToConsole();
	// moving from output to input (excluding the output)
	for (int i = lastHiddenLayerIndex; i >= 0; i--) {
		
	} 
}

void NeuralNetwork::setErrors() {
	int outputLayerIndex = this->layers.size() - 1;
	if (this->target.size() == 0) {
		cerr << "Target is not set for Neural Network!." << endl;
		assert(false);
	}

	if (this->target.size() != this->layers.at(outputLayerIndex)->getNeurons().size()) {
		cerr << "Target is not same size that of the output layer size: " << endl;
		assert(false);
	}

	this->error = 0.0;
	vector<Neuron *> outputNeurons= this->layers.at(outputLayerIndex)->getNeurons();
	for (int i = 0; i < target.size(); i++) {
		double tempErr = (outputNeurons.at(i)->getActivatedVal() - this->target.at(i));
		this->errors.push_back(tempErr);
		this->error += tempErr;
	}

	this->historicalErrors.push_back(this->error);

}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setNeuronVal(i, input.at(i));
	}
}

void NeuralNetwork::printToConsole() {
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "=====================" << endl;
		cout << "LAYER: " << i << endl;
		if (i == 0) {
			Matrix *m = this->layers.at(i)->matrixifyVals();
			m->printToConsole();
		}
		else {
			Matrix *m = this->layers.at(i)->matrixifyActivatedVals();
			m->printToConsole();
		}
		if (i != this->layers.size() - 1) {
			cout << "Weight: " << endl;
			this->getWeightMatrix(i)->printToConsole();
		}
		cout << "=====================" << endl;
	}
}
