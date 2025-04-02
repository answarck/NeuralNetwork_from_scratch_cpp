#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <iostream>
using namespace std;

class Neuron {
public:	
	Neuron(double val); 
	void setVal(double val);
	// Fast Sigmoid Function
	// f(x) = x / (1 + |x|)
	void activate();

	// Derivative of Sigmoid is 
	// f'(x) = f(x) * (1 - f(x))
	void derive();

	// Getter 
	double getVal() { return this->val; }
	double getActivatedVal() { return this->activatedVal; }
	double getDerivedVal() { return this->derivedVal; }

private:
	double val;
	double activatedVal;
	double derivedVal;
};
#endif
