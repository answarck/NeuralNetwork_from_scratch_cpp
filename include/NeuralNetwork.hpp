#ifndef _NEURALNETWORK_HPP_
#define _NEURALNETWORK_HPP_

#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Layer.hpp"
using namespace std;

class NeuralNetwork {
public:	
	NeuralNetwork(vector<int> topology, double learningRate);
	~NeuralNetwork();
	void printToConsole();
	void printInputToConsole();
	void printOutputToConsole();
	void printTargetToConsole();
	void feedForward();
	void backPropogate();
	void setErrors();

	Matrix *getNeuronMatrix(int index) { return this->layers.at(index)->matrixifyVals(); }
	Matrix *getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
	Matrix *getDerivedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyDerivedVals(); }
	Matrix *getWeightMatrix(int index) { return this->weightMatrices.at(index); }
	Matrix *getBiasMatrix(int index) { return this->biasMatrices.at(index) ;}

	void setNeuronValue(int indexLayer, int indexNeuron, double value) { 
		this->layers.at(indexLayer)->setNeuronVal(indexNeuron, value); 
	}
	void setCurrentInput(vector<double> input);
	void setCurrentTarget(vector<double> input) { this->target = input; }
	void setWeightMatrix(int index, Matrix *weightMatrix);
	void setBiasMatrix(int index, Matrix *biasMatrix);

	vector<double> getErrors() { return this->errors; }
	double getError() { return this->error; }
private:
	int 		 topologySize;
	vector<int> 	 topology;
	vector<Layer *>  layers;
	vector<Matrix *> weightMatrices;
	vector<Matrix *> biasMatrices;
	vector<double> input;
	vector<double> target;
	vector<double> errors;
	vector<double> historicalErrors;
	double error;
	double learningRate;


};
#endif
