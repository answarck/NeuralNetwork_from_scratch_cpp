
#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
#include "Neuron.hpp"
using namespace std;

class Matrix {
public:	
	Matrix(int numRows, int numCols, bool isRandom);
	Matrix *transpose();
	vector<double> toVector();
	void printToConsole();
	void setVal(int row, int col, double val) { this->values.at(row).at(col) = val; }
	double getRandNo();
	double getVal(int row, int col) { return this->values.at(row).at(col); }
	int getNumRows() { return this->numRows; }
	int getNumCols() { return this->numCols; }
	Matrix *elementwiseMultiply(Matrix *m);

	// Operator overloading
	Matrix *operator * (Matrix& b);
	Matrix *operator + (Matrix& b);
	Matrix *operator - (Matrix& b);
private:
	int numRows;
	int numCols;

	vector< vector<double> > values;
};
#endif
