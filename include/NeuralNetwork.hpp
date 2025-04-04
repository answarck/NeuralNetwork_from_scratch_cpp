#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Layer.hpp"
using namespace std;

class NeuralNetwork {
public:	
	NeuralNetwork(vector<int> topology);
	void printToConsole();
	void feedForward();
	void setErrors();

	Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
	Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
	Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
	Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); }

	void setNeuronValue(int indexLayer, int indexNeuron, double value) { this->layers.at(indexLayer)->setNeuronVal(indexNeuron, value); }
	void setCurrentInput(vector<double> input);
	void setCurrentTarget(vector<double> input) { this->target = input; }

	vector<double> getErrors() { return this->errors; }
	double getError() { return this->error; }
private:
	int 		 topologySize;
	vector<int> 	 topology;
	vector<Layer *>  layers;
	vector<Matrix *> weightMatrices;
	vector<double> input;
	vector<double> target;
	double error;
	vector<double> errors;
	vector<double> historicalErrors;


};
#endif
