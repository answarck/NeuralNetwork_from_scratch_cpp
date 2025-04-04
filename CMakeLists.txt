cmake_minimum_required(VERSION 3.00)
project(nn_from_scratch)

set(CMAKE_BUILD_TYPE		Debug)
set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -std=c++14 -g")

# Main app
add_executable(
	nn_from_scratch
	src/main.cpp
	src/Neuron.cpp
	src/Matrix.cpp
	src/Layer.cpp
	src/NeuralNetwork.cpp
)

