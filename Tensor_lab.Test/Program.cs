using System;
using Tensor_lab.Layers;
using Tensor_lab.Metrics;
using Tensor_lab.CostFuncs;

namespace Tensor_lab.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Operations K = new Operations();

            //Load array to the tensor
            Tensor x = new Tensor(3, 3);
            x.Load(2, 4, 6, 1, 3, 5, 2, 3, 5);
            x.Print("Load X Values");

            Tensor y = new Tensor(3, 1);
            y.Load(1, 0, 1);
            y.Print("Load Y Values");

            //Create two layers, one with 6 neurons and another with 1
            FullyConnectedLayer fc1 = new FullyConnectedLayer(3, 6, "relu");
            FullyConnectedLayer fc2 = new FullyConnectedLayer(6, 1, "sigmoid");

            //Connect input by passing data from one layer to another
            fc1.Forward(x);
            fc2.Forward(fc1.Output);
            var preds = fc2.Output;
            preds.Print("Predictions");

            //Calculate the mean square error cost between the predicted and expected values
            BaseCost cost = new BinaryCrossEntropy();
            var costValues = cost.Forward(preds, y);
            costValues.Print("BCE Cost");

            //Calculate the mean absolute metric value for the predicted vs expected values
            BaseMetric metric = new BinaryAccuacy();
            var metricValues = metric.Calculate(preds, y);
            metricValues.Print("Acc Metric");

            //Backpropagtion starts here.
            //Calculate gradient of the cost function.
            var grad = cost.Backward(preds, y);

            //Then the FC2 layer by passing the cost function grad into the layer backward function
            fc2.Backward(grad);

            //The grad of the FC2 is stored in the InputGrad property, we need to pass it to the next layer.
            fc1.Backward(fc2.InputGrad);

            //Lets print the parameters for both the layers along with the Grad
            fc1.PrintParams();
            fc2.PrintParams();

            Console.ReadLine();
        }
    }
}
