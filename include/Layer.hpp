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
	void setVal(int index, double value); 
	Matrix *matrixifyVals();
	Matrix *matrixifyActivatedVals();
	Matrix *matrixifyDerivedVals();
private:
	int size;
	vector<Neuron *> neurons;
};
#endif
