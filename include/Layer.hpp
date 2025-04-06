#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "Matrix.hpp"
using namespace std;

class Layer {
public:	
	Layer(int size); 
	void setNeuronVal(int index, double value); 
	double getNeuronVal(int index) { return this->neurons.at(index)->getVal(); }
	vector<Neuron *> getNeurons();
	Matrix *matrixifyVals();
	Matrix *matrixifyActivatedVals();
	Matrix *matrixifyDerivedVals();
	void cleanup();
private:
	int size;
	vector<Neuron *> neurons;
};
#endif
