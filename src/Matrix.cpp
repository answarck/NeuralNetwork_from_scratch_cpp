#include <iostream>
#include <random>
#include <vector>
#include "../include/Matrix.hpp"

Matrix::Matrix(int numRows, int numCols, bool isRandom) {
	this->numRows = numRows;
	this->numCols = numCols;	
	
	for (int i = 0; i < numRows; i++) {
		double r = 0.0;
		vector<double> colVals;
		for (int k = 0; k < numCols; k++) {
			if (isRandom) {
				r = this->getRandNo();
			}
			colVals.push_back(r);
		}		
		this->values.push_back(colVals);
	}
}

double Matrix::getRandNo() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	return dis(gen);
}

void Matrix::printToConsole() {
	for (int i = 0; i < numRows; i++) {
		for (int k = 0; k < numCols; k++) {
			cout << this->values.at(i).at(k) << "\t\t";
		}		
		cout << endl;
	}
}

Matrix *Matrix::transpose() {
	Matrix *m = new Matrix(this->numCols, this->numRows, false);
	for (int i = 0; i < this->numCols; i++) {
		for (int k = 0; k < this->numRows; k++) {
			m->setVal(i, k, this->getVal(k, i));
		}
	}

	return m;
}

