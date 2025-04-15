#include <iostream>
#include <vector>
#include <NeuralNetwork.hpp>

using namespace std;

int main() {
    vector<int> topology = {2, 3, 1};
    NeuralNetwork net(topology, 0.1);

    vector<double> input = {0.5, 0.8};
    
    net.setCurrentInput(input);
    net.feedForward();

    vector<double> target = {0.1};
    net.setCurrentTarget(target);

    net.backPropogate();

    net.saveModel("model.nn");

    cout << "Model saved successfully!" << endl;

    return 0;
}
