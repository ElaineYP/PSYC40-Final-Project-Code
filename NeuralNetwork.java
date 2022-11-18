/*
 * Uses backpropagation to etermine the percent change in the S&P 500 based on macroeconomic variables and the performance of other domestic and foreign stocks
 * 
 * By Elaine Pu, PSYC 40
 * Reference: https://github.com/mattm/simple-neural-network
 * 
 */
import java.util.*;

public class NeuralNetwork {
    private int numInputs;
    private int numHiddenNeurons;
    private int numOutputNeurons;
    private double learningRate;

    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;

    public NeuralNetwork(int numInputs, int numHiddenNeurons, int numOutputNeurons, double learningRate) {
        this.numInputs = numInputs;
        this.numHiddenNeurons = numHiddenNeurons;
        this.numOutputNeurons = numOutputNeurons;
        this.learningRate = learningRate;

        // initialize weights
        ArrayList<ArrayList<Double>> weightMatrixHidden = randomWeights(this.numInputs, this.numHiddenNeurons);
        ArrayList<ArrayList<Double>> weightMatrixOutput = randomWeights(this.numHiddenNeurons, this.numOutputNeurons);
        // build hidden layer and output layer
        outputLayer = new NeuronLayer(this.numOutputNeurons, weightMatrixOutput);
        hiddenLayer = new NeuronLayer(this.numHiddenNeurons, weightMatrixHidden, outputLayer);

    }

    // create matrix of random weights from 0 to 1
    private ArrayList<ArrayList<Double>> randomWeights(int numInputs, int numNeurons) {
        ArrayList<ArrayList<Double>> weightMatrix = new ArrayList<ArrayList<Double>>();
        // loop number of inputs (rows)
        for (int i = 0; i < numInputs; i++) {
            ArrayList<Double> list = new ArrayList<Double>();
            // loop number of neurons/nodes (columns)
            for (int h = 0; h < numNeurons; h++) {
                double num = Math.random();
                list.add(h, num);
            }
            weightMatrix.add(i, list);
        }
        return weightMatrix;
    }

    // given inputs and targets, train the network once
    public void train(ArrayList<Double> inputs, ArrayList<Double> targets) {

        // calculate outputs of hidden neurons
        ArrayList<Double> hiddenLayerOutput = hiddenLayer.getOutputs(inputs);
        // //debug
        // System.out.println("hidden output: \n");
        // System.out.println(hiddenLayerOutput);

        // feed forward to output neurons and calculate output layer neuron outputs
        ArrayList<Double> outputLayerOutput = outputLayer.getOutputs(hiddenLayerOutput);
        // debug
        // System.out.println("output: \n");
        // System.out.println(outputLayerOutput);

        // calculate deltas for output layer: △wij =kδjxi
        ArrayList<ArrayList<Double>> deltasOutputLayer = new ArrayList<ArrayList<Double>>();
        for (int n = 0; n < numOutputNeurons; n++) {
            double error = outputLayer.neurons.get(n).calculateOutputLayerError(targets.get(n));
            ArrayList<Double> deltasOutputNeuron = new ArrayList<Double>();
            // one delta for each input
            for (int i = 0; i < hiddenLayerOutput.size(); i++) {
                double delta = learningRate * error * hiddenLayerOutput.get(i);
                deltasOutputNeuron.add(i, delta);
            }
            // each row represents a neuron's deltas
            deltasOutputLayer.add(n, deltasOutputNeuron);
        }
        // debug
        // System.out.println("deltas: \n");
        // System.out.println(deltasOutputLayer);

        // calculate deltas for hidden layer
        ArrayList<ArrayList<Double>> deltasHiddenLayer = new ArrayList<ArrayList<Double>>();
        for (int n = 0; n < numHiddenNeurons; n++) {
            double error = hiddenLayer.neurons.get(n).calculateHiddenLayerError(outputLayer.neurons, n);
            ArrayList<Double> deltasHiddenNeuron = new ArrayList<Double>();
            // one delta for each input
            for (int i = 0; i < inputs.size(); i++) {
                double delta = learningRate * error * inputs.get(i);
                deltasHiddenNeuron.add(i, delta);
            }
            // each row represents a neuron's deltas
            deltasHiddenLayer.add(n, deltasHiddenNeuron);
        }

        // change hidden layer weights
        for (int n = 0; n < numHiddenNeurons; n++) {
            for (int w = 0; w < hiddenLayer.neurons.get(n).weights.size(); w++) {
                double delta = deltasHiddenLayer.get(n).get(w);
                double newWeight = hiddenLayer.neurons.get(n).weights.get(w) + delta;
                hiddenLayer.neurons.get(n).weights.set(w, newWeight);
            }
        }

        // change output layer weights
        for (int n = 0; n < numOutputNeurons; n++) {
            for (int w = 0; w < outputLayer.neurons.get(n).weights.size(); w++) {
                double delta = deltasOutputLayer.get(n).get(w);
                double newWeight = outputLayer.neurons.get(n).weights.get(w) + delta;
                outputLayer.neurons.get(n).weights.set(w, newWeight);
            }
        }
    }

    public double totalError(ArrayList<ArrayList<Double>> inputSets, ArrayList<ArrayList<Double>> targetSet) {
        double total = 0.0;
        // loop each set of inputs
        for (int s = 0; s < inputSets.size(); s++) {
            // calculate error for each set with newly trained weights
            for (int n = 0; n < outputLayer.numNeurons; n++) {
                // recalculate output for each input set
                ArrayList<Double> hiddenLayerOutput = hiddenLayer.getOutputs(inputSets.get(s));
                // feed forward to output neurons and calculate output layer neuron outputs
                outputLayer.getOutputs(hiddenLayerOutput);
                // difference between target and output
                double difference = Math.pow(targetSet.get(s).get(n) - outputLayer.neurons.get(n).output, 2) / 2;
                // sum differences across output neurons
                total += difference;
            }
        }
        return total;
    }
    
    //Uses existing weights to calculate output and prints the output vs target data
    public void test(ArrayList<ArrayList<Double>> inputSets, ArrayList<ArrayList<Double>> targetSet){
        double total = 0;
        for (int s = 0; s < inputSets.size(); s++) {
            // calculate error for each set with newly trained weights
            for (int n = 0; n < outputLayer.numNeurons; n++) {
                // recalculate output for each input set
                ArrayList<Double> hiddenLayerOutput = hiddenLayer.getOutputs(inputSets.get(s));
                // feed forward to output neurons and calculate output layer neuron outputs
                outputLayer.getOutputs(hiddenLayerOutput);
                // difference between target and output
                double difference = Math.pow(targetSet.get(s).get(n) - outputLayer.neurons.get(n).output, 2) / 2;
                // sum differences across output neurons
                total += difference;
                //print output, expected output, and MSE of the between them
                System.out.println("Output: " + outputLayer.neurons.get(n).output+ " Target: "+ targetSet.get(s).get(n) + " MSE: " + difference);
            }
        }
        //print total error for the entire test set
        System.out.println("Final error: "+ total);
    }

    //print current weights for hidden layer and output layer
    public void printWeights(){
        System.out.println("Hidden layer weights: ");
        for (int n = 0; n< hiddenLayer.neurons.size(); n++){
            System.out.println(hiddenLayer.neurons.get(n).weights);
        }
        System.out.println("Output layer weights: ");
        for (int n = 0; n< outputLayer.neurons.size(); n++){
            System.out.println(outputLayer.neurons.get(n).weights);
        }
    }
    //prints for debugging
    @Override
    public String toString() {
        String strHidden = "Hidden layer: \n";
        for (int n = 0; n < numHiddenNeurons; n++) {
            strHidden += hiddenLayer.neurons.get(n).neuronToString();
            strHidden += "\n";
        }
        String strOutput = "Output layer: \n";
        for (int n = 0; n < numOutputNeurons; n++) {
            strOutput += outputLayer.neurons.get(n).neuronToString();
            strOutput += "\n";
        }
        return strHidden + strOutput;
    }

    private class Neuron {
        ArrayList<Double> weights;
        double net;
        double output;
        double error;

        // neuron in output layer
        Neuron(ArrayList<Double> weights) {
            this.weights = weights;
        }

        // neuron in hidden layer
        Neuron(ArrayList<Double> weights, int layer, ArrayList<Neuron> neurons) {
            this.weights = weights;
        }

        // calculate net of neuron based on given inputs using performance rule yj
        // =f(∑(wij xii))
        // needs to be called every time input changes before calling other functions
        double calculateOutput(ArrayList<Double> inputs) {
            double net = 0.0;
            // calculate net by multiplying each input with corresponding weights and
            // summing them
            for (int i = 0; i < inputs.size(); i++) {
                net += inputs.get(i) * weights.get(i);
            }
            this.net = net;
            output = calculateOutputFunc();
            // //debug
            // System.out.print("inputs calcOut: ");
            // System.out.println(inputs);
            // System.out.print("weights calcOut: ");
            // System.out.println(weights);
            // System.out.print("net calcOut: ");
            // System.out.println(net);
            // System.out.print("out calcOut: ");
            // System.out.println(output);
            return output;
        }

        // f(net) = 1/(1+e^(-net))
        double calculateOutputFunc() {
            return 1 / (1 + Math.exp(-net));
        }

        // f'(net) = (e^(-net))/((e^(-net) + 1))^2)
        double calculateFuncPrime() {
            double numerator = Math.exp(-net);
            double denominator = Math.pow((Math.exp(-net) + 1), 2);
            return numerator / denominator;
        }

        // given target, calculate error of neuron in output layer = (tj−yj)f'(netj)
        double calculateOutputLayerError(double target) {
            double error = (target - output);
            error = error * calculateFuncPrime();
            this.error = error;
            // debug
            // System.out.print("output layer error: ");
            // System.out.println(this.error);
            return this.error;
        }

        // calculate error of neuron in hidden layer =(∑δkwjk)f'(netj)
        double calculateHiddenLayerError(ArrayList<Neuron> neurons, int layer) {
            double total = 0;
            // loop through every neuron this neuron outputs to
            for (int n = 0; n < neurons.size(); n++) {
                // get the weights effected by this neuron's outputs
                // get the effected neuron's error
                total += (neurons.get(n).weights.get(layer)) * (neurons.get(n).error);
            }
            error = total * calculateFuncPrime();
            // debug
            // System.out.print("hidden layer error: ");
            // System.out.println(this.error);
            return error;
        }

        //prints the neuron for debugging
        String neuronToString() {
            String str = "Neuron: \n";
            str += "Weights: ";
            str += weights;
            str += "\n";
            str += "Output: ";
            str += output;
            str += "\n";

            return str;
        }
    }

    private class NeuronLayer {
        int numNeurons;
        ArrayList<Neuron> neurons = new ArrayList<Neuron>();
        ArrayList<ArrayList<Double>> weightMatrix = new ArrayList<ArrayList<Double>>();

        // output layer
        NeuronLayer(int numNeurons, ArrayList<ArrayList<Double>> weightMatrix) {
            this.numNeurons = numNeurons;
            this.weightMatrix = weightMatrix;
            for (int n = 0; n < numNeurons; n++) {
                // new array to hold the new neuron's weights
                ArrayList<Double> weights = new ArrayList<Double>();
                // w correspond to each input, n correspond to each neuron
                for (int w = 0; w < this.weightMatrix.size(); w++) {
                    // assign one weight to each input
                    weights.add(this.weightMatrix.get(w).get(n));
                }
                neurons.add(n, new Neuron(weights));
            }
        }

        // hidden layer
        NeuronLayer(int numNeurons, ArrayList<ArrayList<Double>> weightMatrix, NeuronLayer nextLayer) {
            this.numNeurons = numNeurons;
            this.weightMatrix = weightMatrix;
            for (int n = 0; n < numNeurons; n++) {
                // new array to hold the new neuron's weights
                ArrayList<Double> weights = new ArrayList<Double>();
                // w correspond to each input, n correspond to each neuron
                for (int w = 0; w < this.weightMatrix.size(); w++) {
                    // assign one weight to each input
                    weights.add(this.weightMatrix.get(w).get(n));
                }
                // neuron at index0 is at layer0
                neurons.add(new Neuron(weights, n, nextLayer.neurons));
            }
        }

        // pass inputs into each neuron and get the output of each neuron in this layer
        ArrayList<Double> getOutputs(ArrayList<Double> inputs) {
            ArrayList<Double> outputs = new ArrayList<Double>();
            for (int n = 0; n < neurons.size(); n++) {
                neurons.get(n).calculateOutput(inputs);
                outputs.add(n, neurons.get(n).output);
            }
            return outputs;
        }
    }

    public static void main(String[] args) {
        int numInputs = 5;
        int numHiddenNeurons = 2;
        int numOutputNeurons = 1;
        double learningRate = 1;

        NeuralNetwork network = new NeuralNetwork(numInputs, numHiddenNeurons, numOutputNeurons,
                learningRate);

        /*
         * simple test
         * ArrayList<Double> inputs1 = new ArrayList<>();
         * inputs1.add(0.3);
         * inputs1.add(0.1);
         * ArrayList<Double> targets1 = new ArrayList<>();
         * targets1.add(0.2);
         * targets1.add(0.4);
         * 
         * ArrayList<Double> inputs2 = new ArrayList<>();
         * inputs2.add(0.1);
         * inputs2.add(0.8);
         * ArrayList<Double> targets2 = new ArrayList<>();
         * targets2.add(0.2);
         * targets2.add(0.1);
         * 
         * ArrayList<ArrayList<Double>> inputSet = new ArrayList<ArrayList<Double>>();
         * inputSet.add(inputs1);
         * inputSet.add(inputs2);
         * 
         * ArrayList<ArrayList<Double>> targetSet = new ArrayList<ArrayList<Double>>();
         * targetSet.add(targets1);
         * targetSet.add(targets2);
         */

        ReadFile sp500File = new ReadFile("data/sp500");
        ArrayList<Double> sp500 = sp500File.fileToList();

        ReadFile moneySupplyFile = new ReadFile("data/moneySupply");
        ArrayList<Double> moneySupply = moneySupplyFile.fileToList();
        // degug
        // System.out.println(moneySupplyTrain);

        ReadFile gdpFile = new ReadFile("data/gdp");
        ArrayList<Double> gdp = gdpFile.fileToList();

        ReadFile nasdaqFile = new ReadFile("data/nasdaq");
        ArrayList<Double> nasdaq = nasdaqFile.fileToList();
        // degug
        // System.out.println(nasdaqTrain);

        ReadFile nikkei225File = new ReadFile("data/nikkei225");
        ArrayList<Double> nikkei225 = nikkei225File.fileToList();
        // degug
        // System.out.println(nikkei225Train);

        ReadFile yieldFile = new ReadFile("data/10yearYield");
        ArrayList<Double> yield = yieldFile.fileToList();
        // degug
        // System.out.println (yieldTrain);

        // builds a matrix of inputs
        ArrayList<ArrayList<Double>> inputMatrix = new ArrayList<ArrayList<Double>>();
        // each list should be the same size
        inputMatrix.add(moneySupply);
        inputMatrix.add(gdp);
        inputMatrix.add(nasdaq);
        inputMatrix.add(nikkei225);
        inputMatrix.add(yield);

        // builds the inputSet from the inputMatrix
        ArrayList<ArrayList<Double>> inputSet = new ArrayList<ArrayList<Double>>();
        // loop each column of inputMatrix, this is how many set of inputs there will be
        for (int col = 0; col < moneySupply.size(); col++) {
            ArrayList<Double> inputs = new ArrayList<>();
            // loop each row of inputMatrix, this is how many inputs there are in each set
            for (int row = 0; row < inputMatrix.size(); row++) {
                inputs.add(inputMatrix.get(row).get(col));
            }
            inputSet.add(inputs);
        }

        ArrayList<ArrayList<Double>> targetSet = new ArrayList<ArrayList<Double>>();
        for (int n = 0; n < sp500.size(); n++) {
            ArrayList<Double> targets = new ArrayList<>();
            targets.add(sp500.get(n));
            targetSet.add(targets);
        }

        // randomly select testing and training sets
        //training set of 200 sets of inputs and target
        ArrayList<ArrayList<Double>> inputTrainSet = new ArrayList<>();
        ArrayList<ArrayList<Double>> targetTrainSet = new ArrayList<>();
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < targetSet.size(); i++) {
            list.add(i);
        }
        //random indexes
        Collections.shuffle(list);
        //System.out.println(list);
        //half of the data is put into training set
        for (int i = 0; i < 183; i++) {
            inputTrainSet.add(i, inputSet.get(i));
            targetTrainSet.add(i, targetSet.get(i));
        }
        //add remaining data to test set
        ArrayList<ArrayList<Double>> inputTestSet = new ArrayList<>();
        ArrayList<ArrayList<Double>> targetTestSet = new ArrayList<>();
        for (int i = 183; i < list.size(); i++) {
            inputTestSet.add(inputSet.get(i));
            targetTestSet.add(targetSet.get(i));
        }

        
        //print initial weights
        System.out.println("Initial weights: ");
        network.printWeights();
        //print initial error with test sets
        System.out.println("Initial error: "+network.totalError(inputTestSet, targetTestSet));
        
        //train the network with train sets
        System.out.println("Train");
        for (int i = 0; i < 700; i++) {
            // pick a random set of inputs and targets
            ArrayList<Double> inputs = new ArrayList<>();
            ArrayList<Double> targets = new ArrayList<>();
            int random = (int) (Math.random() * (Integer.MAX_VALUE)) % inputTrainSet.size();
            inputs = inputTrainSet.get(random);
            targets = targetTrainSet.get(random);
            // train with this set
            network.train(inputs, targets);
            // debug
            // System.out.println(network);
            // calculate total error
            System.out.println(network.totalError(inputTrainSet, targetTrainSet));
        }

        //test the network
        System.out.println("Test");
        network.test(inputTestSet, targetTestSet);
        //print final weights
        System.out.println("Final weights: ");
        network.printWeights();
    }
}
