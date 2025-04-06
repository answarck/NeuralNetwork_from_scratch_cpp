#include <vector>
#include <cassert>
#include <cmath>
#include "../include/NeuralNetwork.hpp"
#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topologySize = topology.size();
	this->topology = topology;

	for (int i = 0; i < topology.size(); i++) {
		Layer *l = new Layer(topology.at(i));
		Matrix *b = new Matrix(1, topology.at(i), false);
		this->biasMatrices.push_back(b);
		this->layers.push_back(l);
	}

	for (int i = 0; i < topology.size() - 1; i++) {
		Matrix *m = new Matrix(topology.at(i + 1), topology.at(i), true);
		this->weightMatrices.push_back(m);
	}

}

NeuralNetwork::~NeuralNetwork() {
	for (int i = 0; i < this->layers.size(); i++) {
		this->layers.at(i)->cleanup();
		delete this->biasMatrices.at(i);
		delete layers.at(i);
	}
	for (int i = 0; i < this->weightMatrices.size(); i++) {
		delete this->weightMatrices.at(i);
	}
}

void NeuralNetwork::feedForward() {
	for (int i = 0; i < (this->layers.size() - 1); i++) {
		Matrix *a;

		if (i != 0) {
			a = this->getActivatedNeuronMatrix(i);
		}	
		else {
			a = this->getNeuronMatrix(i);
		}

		Matrix *b = this->getWeightMatrix(i)->transpose();

		Matrix *c = *a * *b;

		for (int k = 0; k < c->getNumCols(); k++) {
			this->setNeuronValue(i + 1, k, c->getVal(0, k));
		}

		delete a;
		delete b;
		delete c;
	} 

	this->setErrors();
}

void NeuralNetwork::backPropogate() {
	// Hidden -> Output
	int outputLayerIndex = this->layers.size() - 1;
	int lastHiddenLayerIndex = outputLayerIndex - 1;
	Matrix *output = this->layers.at(outputLayerIndex)->matrixifyVals();

	Matrix *target = new Matrix(1, output->getNumCols(), false);
	for (int i = 0; i < output->getNumCols(); i++) {
		target->setVal(0, i, this->target.at(i));
	}

	Matrix *derivedVals = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
	Matrix *delta = *output - *target;
	Matrix *deltaT = delta->transpose();

	Matrix *outputLayerDelta = delta->elementwiseMultiply(derivedVals);
	Matrix *outputLayerDeltaT = outputLayerDelta->transpose();

	Matrix *activatedVals = this->layers.at(lastHiddenLayerIndex)->matrixifyActivatedVals();
	Matrix *weights = this->getWeightMatrix(lastHiddenLayerIndex);
	Matrix *biases = this->getBiasMatrix(outputLayerIndex);
	
	// Gradients calculated (OUTPUT)
	Matrix *gradient = *outputLayerDeltaT * *activatedVals;


	// Updating Bias and Weights
	Matrix *updatedWeights = *this->getWeightMatrix(lastHiddenLayerIndex) - *gradient;
	Matrix *updatedBiases = *this->getBiasMatrix(outputLayerIndex) - *delta;
	this->setWeightMatrix(lastHiddenLayerIndex, updatedWeights);
	this->setBiasMatrix(outputLayerIndex, updatedBiases);

	// cleanup from HIDDEN->OUTPUT
	delete outputLayerDelta;
	delete outputLayerDeltaT;
	delete derivedVals;
	delete activatedVals;
	delete gradient;
	delete target;
	delete output;

	// Input to hidden and hidden to hidden
	for (int i = lastHiddenLayerIndex - 1; i >= 0; i--) {
		weights = this->getWeightMatrix(i);
		biases = this->getBiasMatrix(i + 1);
		derivedVals = this->layers.at(i + 1)->matrixifyDerivedVals(); 
		
		Matrix *vals;
		if (i == 0) {
			vals = this->layers.at(i)->matrixifyVals();
		}
		else {
			vals = this->layers.at(i)->matrixifyActivatedVals();
		}
		
		Matrix *dA  = *weights * *deltaT;
		Matrix *dAT = dA->transpose();	

		// DELTA MATRIX cleanup
		delete delta;
		delete deltaT;

		delta = dAT->elementwiseMultiply(derivedVals);
		deltaT = delta->transpose();

		gradient = *deltaT * *vals;

		updatedWeights = *weights - *gradient;
		updatedBiases = *biases - *delta;

		this->setWeightMatrix(i, updatedWeights);
		this->setBiasMatrix(i + 1, updatedBiases);

		// Input/Hidden -> Hidden Memory cleanup
		// Dont't delete updatedWeights
		delete dA;
		delete dAT;
		delete gradient;
		delete derivedVals;
		delete vals;
		
	}
	
	delete delta;
	delete deltaT;
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
		double tempErr = 0.5 * pow(outputNeurons.at(i)->getActivatedVal() - this->target.at(i), 2);
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

void NeuralNetwork::setWeightMatrix(int index, Matrix *weightMatrix) {
	delete this->weightMatrices.at(index);
	this->weightMatrices.at(index) = weightMatrix; 
}

void NeuralNetwork::setBiasMatrix(int index, Matrix *biasMatrix) {
	delete this->biasMatrices.at(index);
	this->biasMatrices.at(index) = biasMatrix; 
}

void NeuralNetwork::printInputToConsole() {
	cout << "==========" << endl;
	cout << "INPUT: " << endl;
	Matrix *m = this->layers.at(0)->matrixifyVals();
	m->printToConsole();
	delete m;
}

void NeuralNetwork::printOutputToConsole() {
	cout << "==========" << endl;
	cout << "OUTPUT: " << endl;
	Matrix *m = this->layers.at(this->layers.size() - 1)->matrixifyVals();
	m->printToConsole();
	delete m;
}

void NeuralNetwork::printTargetToConsole() {
	cout << "==========" << endl;
	cout << "TARGET: " << endl;
	for (int i = 0; i < this->target.size(); i++) {
		cout << this->target.at(i) << "\t"; 
	}
	cout << endl;
}
void NeuralNetwork::printToConsole() {
	for (int i = 0; i < this->layers.size(); i++) {
		cout << "=====================" << endl;
		cout << "LAYER: " << i << endl;
		Matrix *m;
		if (i == 0) {
			m = this->layers.at(i)->matrixifyVals();
			m->printToConsole();
		}
		else {
			m = this->layers.at(i)->matrixifyActivatedVals();
			m->printToConsole();
		}
		if (i != this->layers.size() - 1) {
			cout << "Weight: " << endl;
			Matrix *wM = this->getWeightMatrix(i);
			wM->printToConsole();
			cout << "________________" << endl;
		}
		if (i != 0) {
			cout << "Bias: " << endl;
			Matrix *bM = this->getBiasMatrix(i);
			bM->printToConsole();
		}
		cout << "=====================" << endl;

		delete m;
	}
}
