#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

Layer::Layer(int size) {
	for (int i = 0; i < size ; i++) {
		Neuron *n = new Neuron(0.00);
		this->neurons.push_back(n);	
	}
	this->size = size;
}

void Layer::setNeuronVal(int index, double value) {
	this->neurons.at(index)->setVal(value);
}

Matrix *Layer::matrixifyVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(0, i, neurons.at(i)->getVal());
	}
	
	return m;
}

Matrix *Layer::matrixifyActivatedVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(0, i, neurons.at(i)->getActivatedVal());
	}
	
	return m;
}

Matrix *Layer::matrixifyDerivedVals() {
	Matrix *m = new Matrix(1, this->neurons.size(), false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(0, i, neurons.at(i)->getDerivedVal());
	}
	
	return m;
}
