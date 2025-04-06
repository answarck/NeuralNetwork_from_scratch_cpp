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
		this->layers.push_back(l);
	}

	for (int i = 0; i < topology.size() - 1; i++) {
		Matrix *m = new Matrix(topology.at(i + 1), topology.at(i), true);
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

		Matrix *c = *a * *b->transpose();

		for (int k = 0; k < c->getNumCols(); k++) {
			this->setNeuronValue(i + 1, k, c->getVal(0, k));
		}
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

	Matrix *derivedVal = this->layers.at(outputLayerIndex)->matrixifyDerivedVals();
	Matrix *delta = *output - *target;

	Matrix *outputLayerDelta = delta->elementwiseMultiply(derivedVal)->transpose();

	Matrix *activatedVals = this->layers.at(lastHiddenLayerIndex)->matrixifyActivatedVals();
	Matrix *weights = this->getWeightMatrix(lastHiddenLayerIndex);
	
	// Gradients calculated (OUTPUT)
	Matrix *gradient = *outputLayerDelta * *activatedVals;


	// Updating Bias(WILL UPDATE) and Weights
	Matrix *updatedWeights = *this->getWeightMatrix(lastHiddenLayerIndex) - *gradient;
	this->setWeightMatrix(lastHiddenLayerIndex, updatedWeights);

	// Input to hidden and hidden to hidden
	for (int i = lastHiddenLayerIndex - 1; i >= 0; i--) {
		weights = this->getWeightMatrix(i);
		derivedVal = this->layers.at(i + 1)->matrixifyDerivedVals(); 
		
		Matrix *vals;
		if (i == 0) {
			vals = this->layers.at(i)->matrixifyVals();
		}
		else {
			vals = this->layers.at(i)->matrixifyActivatedVals();
		}
		
		delta = ((*weights * *delta->transpose())->transpose())->elementwiseMultiply(derivedVal);
		gradient = *delta->transpose() * *vals;
		
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

void NeuralNetwork::printInputToConsole() {
	cout << "==========" << endl;
	cout << "INPUT: " << endl;
	this->layers.at(0)->matrixifyVals()->printToConsole();
}

void NeuralNetwork::printOutputToConsole() {
	cout << "==========" << endl;
	cout << "OUTPUT: " << endl;
	this->layers.at(this->layers.size() - 1)->matrixifyVals()->printToConsole();
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
