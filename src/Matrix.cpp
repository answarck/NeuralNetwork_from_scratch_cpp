#include <iostream>
#include <random>
#include <vector>
#include <cassert>

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
			cout << this->values.at(i).at(k) << "\t";
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

Matrix *Matrix::operator+(Matrix& b) {
	if (this->getNumRows() != b.getNumRows() || this->getNumCols() != b.getNumCols()) {
		std::cerr << "Rows and Column sizes mismatch: " << std::endl;
		assert(false);
	}

	Matrix *m = new Matrix(this->getNumRows(), this->getNumCols(), false);

	for (int i = 0; i < this->getNumRows(); i++) {
		for (int k = 0; k < this->getNumCols(); k++) {
			m->setVal(i,
				  k, 
				  this->getVal(i, k) + b.getVal(i , k));
		}
	
	}

	return m;
}

Matrix *Matrix::operator-(Matrix& b) {
	if (this->getNumRows() != b.getNumRows() || this->getNumCols() != b.getNumCols()) {
		std::cerr << "Rows and Column sizes mismatch: " << std::endl;
		assert(false);
	}

	Matrix *m = new Matrix(this->getNumRows(), this->getNumCols(), false);

	for (int i = 0; i < this->getNumRows(); i++) {
		for (int k = 0; k < this->getNumCols(); k++) {
			m->setVal(i,
				  k, 
				  this->getVal(i, k) - b.getVal(i , k));
		}
	
	}

	return m;
}

Matrix *Matrix::operator*(Matrix& b) {
    if (this->getNumCols() != b.getNumRows()) {
        std::cerr << "A_cols: " << this->getNumCols() << " B_rows: " << b.getNumRows() << std::endl;
        assert(false);
    }

    Matrix *c = new Matrix(this->getNumRows(), b.getNumCols(), false);

    for (int i = 0; i < this->getNumRows(); i++) {
        for (int k = 0; k < b.getNumCols(); k++) {
            for (int l = 0; l < this->getNumCols(); l++) { 
                double v = this->getVal(i, l) * b.getVal(l, k);
                double nv = c->getVal(i, k) + v;
                c->setVal(i, k, nv);
            }
        }
    }

    return c;
}

Matrix *Matrix::elementwiseMultiply(Matrix *m) {
	if (m->getNumRows() != this->getNumRows() || m->getNumCols() != this->getNumCols()) {
		cerr << "Dimensions mismatch for the matrix: " << endl;
		assert(false);
	}

	Matrix *temp = new Matrix(m->getNumRows(), m->getNumCols(), false);

	for (int i = 0; i < m->getNumRows(); i++) {
		for (int k = 0; k < m->getNumCols(); k++) {
			temp->setVal(i, k, this->getVal(i, k) * m->getVal(i, k));
		}
	}

	return temp;
}

vector<double> Matrix::toVector() {
	vector<double> v;
	for (int i = 0; i < this->values.size(); i++) {
		for (int k = 0; k < this->values.at(i).size(); k++) {
			v.push_back(values.at(i).at(k));
		}
	}

	return v;
}
