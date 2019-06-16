//
//  AppDelegate.swift
//  ArtificialNeuronNetwork
//
//  Created by Anita Agrawal on 12/06/19.
//  Copyright © 2019 Anita Agrawal. All rights reserved.
//

import UIKit
import Foundation

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        //Brain will train the Neural network with provided inputs and outputs.
        //For the simplicity of the demo, I've used logical operators
        let brain = Brain(dataSet: .xor)
        
        print("Input1 = 0, Input2 = 0, Output = \(brain.Train(inputs: [0, 0, 0], desiredOutput: 1))")
        print("Input1 = 0, Input2 = 1, Output = \(brain.Train(inputs: [0, 1, 0], desiredOutput: 0))")
        print("Input1 = 1, Input2 = 0, Output = \(brain.Train(inputs: [1, 0, 0], desiredOutput: 0))")
        print("Input1 = 1, Input2 = 1, Output = \(brain.Train(inputs: [1, 1, 1], desiredOutput: 1))")

        return true
    }

}

enum LogicalOperator {
    case or, and, xor, xnor
    
    func getDataSet() -> [DataSet] {
        switch self {
        case .or:
            return initializeDataSetForOR()
        case .and:
            return initializeDataSetForAND()
        case .xor:
            return initializeDataSetForXORThreeInputs()
        case .xnor:
            return initializeDataSetForXNOR()
        }
    }
    
    func initializeDataSetForOR() -> [DataSet] {
        var dataSet = [DataSet]()
        dataSet.append(DataSet(inputs: [0,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [0,1], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,1], desiredOutput: 1))
        return dataSet
    }
    
    func initializeDataSetForAND() -> [DataSet] {
        var dataSet = [DataSet]()
        dataSet.append(DataSet(inputs: [0,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [0,1], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [1,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [1,1], desiredOutput: 1))
        return dataSet
    }
    
    func initializeDataSetForXOR() -> [DataSet] {
        var dataSet = [DataSet]()
        dataSet.append(DataSet(inputs: [0,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [0,1], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,1], desiredOutput: 0))
        return dataSet
    }
    
    func initializeDataSetForXORThreeInputs() -> [DataSet] {
        var dataSet = [DataSet]()
        dataSet.append(DataSet(inputs: [0,0,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [0,0,1], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [0,1,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [0,1,1], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,0,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,0,1], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,1,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [1,1,1], desiredOutput: 0))
        return dataSet
    }
    
    func initializeDataSetForXNOR() -> [DataSet] {
        var dataSet = [DataSet]()
        dataSet.append(DataSet(inputs: [0,0], desiredOutput: 1))
        dataSet.append(DataSet(inputs: [0,1], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [1,0], desiredOutput: 0))
        dataSet.append(DataSet(inputs: [1,1], desiredOutput: 1))
        return dataSet
    }
}

struct DataSet {
    let inputs: [Double]
    let desiredOutput: Double
}

class Neuron {
    var inputs: [Double] = []
    var weights: [Double] = []
    var bias: Double = 0
    var output: Double = 0
    var errorGradient: Double = 0
    var noOfInputs: Int
    
    init(numberOfInputs: Int) {
        noOfInputs = numberOfInputs
        initializeWeights()
        initializeBias()
    }
    
    func initializeWeights() {
        weights = []
        for _ in 0 ..< noOfInputs {
            weights.append(Double.random(in: -1 ... 1))
        }
    }
    
    func initializeBias() {
        bias = Double.random(in: -1 ... 1)
    }
    
    //sum of product of input & weight and subtracted by bias
    func calculateDotProductOutput(inputs: [Double]) -> Double {
        
        var tempOutput = 0.0
        for i in 0 ..< inputs.count {
            tempOutput += (inputs[i] * weights[i])
        }
        //Bias can be aded as well instead of subtracted
        tempOutput -= bias
        return tempOutput
    }
    
    //For activation function I'm using sigmoid. can be used Step, tanh or other functions
    func activationFunction(calculatedOutput: Double) -> Double {
        return sigmoid(z: calculatedOutput)
    }
    
    func sigmoid(z: Double) -> Double {
        let k = exp(z)
        return k / (1.0 + k)
    }
    
    //This updates the output of the neuron which will be input to the next layer neuron
    func updateOutput() {
        let calculatedOutput = calculateDotProductOutput(inputs:inputs)
        output = activationFunction(calculatedOutput: calculatedOutput)
    }
}

//Layer could be hidden layer or output layer.
class Layer {
    var neurons = [Neuron]()
    var noOfInputsPerNeuron: Int
    var noOfNeurons: Int
    
    init(numberOfInputs: Int, numberOfNeurons: Int) {
        noOfInputsPerNeuron = numberOfInputs
        noOfNeurons = numberOfNeurons
        for _ in 0 ..< numberOfNeurons {
            let neuron = Neuron(numberOfInputs: noOfInputsPerNeuron)
            neurons.append(neuron)
        }
    }
}

//This is Artifical Neural Network class, which will consist the all the layers present.
//This class is the heart of the algorithm
class ANN {
    var layers = [Layer]()
    var noOfHidderLayers: Int
    var noOfNeuronsPerHL: Int
    var noOfInputs: Int
    var noOfOutput: Int
    var alpha = 0.8
    
    //Init method will initialize the NN with random weights and bias for neurons, and create all the layers and neurons.
    init(numberOfHiddenLayer: Int = 2, numberOfNeuronsPerHL: Int = 3, numberOfInputs: Int = 2, numberOfOutputs: Int = 1, alphaValue:Double = 0.8) {
        noOfHidderLayers = numberOfHiddenLayer
        noOfNeuronsPerHL = numberOfNeuronsPerHL
        noOfInputs = numberOfInputs
        noOfOutput = numberOfOutputs
        alpha = alphaValue
        layers.append(Layer(numberOfInputs: noOfInputs, numberOfNeurons: noOfNeuronsPerHL))
        for _ in 0 ..< (noOfHidderLayers - 1) {
            let layer = Layer(numberOfInputs: noOfNeuronsPerHL, numberOfNeurons: noOfNeuronsPerHL)
            layers.append(layer)
        }
        layers.append(Layer(numberOfInputs: noOfNeuronsPerHL, numberOfNeurons: noOfOutput))
    }
    
    //This will update the outputs of the neurons for the given inputs
    func trainModel(inputs: [Double]) -> [Double] {
        var outputs = [Double]()
        var neuronInput = inputs
        for layer in layers {
            for neuron in layer.neurons {
                neuron.inputs = neuronInput
                neuron.updateOutput()
                outputs.append(neuron.output)
            }
            neuronInput = outputs
            outputs = []
        }
        return neuronInput
    }
    
    func go(inputValues: [Double], desiredOutput: [Double]) -> [Double]? {
        if inputValues.count != noOfInputs {
            print("Number of inputs should be same")
            return nil
        }
        let calculatedOutput = trainModel(inputs: inputValues)
        updateWeightsAndBias(calculatedOutput: calculatedOutput, desiredOutput: desiredOutput)
        return calculatedOutput
    }
    
    //This method will update Weights, bias, and error of each neuron
    func updateWeightsAndBias(calculatedOutput: [Double], desiredOutput: [Double]) {
        var error = 0.0
        
        for i in (0 ..< layers.count).reversed() {
            for j in 0 ..< layers[i].noOfNeurons {
                if i == (layers.count - 1) {//output layer
                    error = desiredOutput[j] - calculatedOutput[j];
                    layers[i].neurons[j].errorGradient = calculatedOutput[j] * (1-calculatedOutput[j]) * error
                } else {//Hidden layer
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1-layers[i].neurons[j].output)
                    var grandErrorSum = 0.0
                    for neuron in layers[i + 1].neurons {
                        grandErrorSum += neuron.errorGradient * neuron.weights[j]
                    }
                    layers[i].neurons[j].errorGradient *= grandErrorSum
                }
                for k in 0 ..< layers[i].neurons[j].noOfInputs {
                    if i == (layers.count - 1) {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error
                    } else {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient
                    }
                }
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient
            }
        }
    }
}

//This will initialize neural network and train the model for the input data set.
class Brain {
    var ann = ANN(numberOfHiddenLayer: 2, numberOfNeuronsPerHL: 3, numberOfInputs: 3, numberOfOutputs: 1)
    var result = [Double]()
    init(dataSet: LogicalOperator) {
        for _ in 0 ..< 3000
        {
            for dataSet in dataSet.getDataSet() {
                result = ann.go(inputValues: dataSet.inputs, desiredOutput: [dataSet.desiredOutput]) ?? []
            }
        }
    }
    
    func Train(inputs: [Double], desiredOutput: Double) -> [Double] {
        return ann.trainModel(inputs: inputs)
    }
}
