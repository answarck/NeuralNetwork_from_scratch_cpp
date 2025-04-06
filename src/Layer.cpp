#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

Layer::Layer(int size) {
	for (int i = 0; i < size ; i++) {
		Neuron *n = new Neuron(0.00);
		this->neurons.push_back(n);	
	}
	this->size = size;
}

void Layer::cleanup() {
	for (int i = 0; i < this->neurons.size(); i++) {
		delete this->neurons.at(i);
	}
}

void Layer::setNeuronVal(int index, double value) {
	this->neurons.at(index)->setVal(value);
}

vector<Neuron *> Layer::getNeurons() {
	vector<Neuron *> temp;
	for (int i = 0; i < this->neurons.size(); i++) {
		temp.push_back(this->neurons.at(i));
	}

	return temp;
}

Matrix *Layer::matrixifyVals() {
	Matrix *m = new Matrix(this->neurons.size(), 1, false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(i, 0, neurons.at(i)->getVal());
	}
	
	return m;
}

Matrix *Layer::matrixifyActivatedVals() {
	Matrix *m = new Matrix(this->neurons.size(), 1, false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(i, 0, neurons.at(i)->getActivatedVal());
	}
	
	return m;
}

Matrix *Layer::matrixifyDerivedVals() {
	Matrix *m = new Matrix(this->neurons.size(), 1, false);
	for (int i = 0; i < this->neurons.size(); i++) {
		m->setVal(i, 0, neurons.at(i)->getDerivedVal());
	}
	
	return m;
}
