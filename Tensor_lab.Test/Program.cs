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
            y.Load(20, 15, 15);
            y.Print("Load Y Values");

            K.Mean(y, 0).Print();

            //Create two layers, one with 6 neurons and another with 1
            FullyConnectedLayer fc1 = new FullyConnectedLayer(3, 6, "relu");
            FullyConnectedLayer fc2 = new FullyConnectedLayer(6, 1, "relu");

            //Connect input by passing data from one layer to another
            fc1.Forward(x);
            fc2.Forward(fc1.Output);
            var preds = fc2.Output;
            preds.Print("Predictions");

            //Calculate the mean square error cost between the predicted and expected values
            BaseCost cost = new MeanSquaredError();
            var costValues = cost.Forward(preds, y);
            costValues.Print("MSE Cost");

            //Calculate the mean absolute metric value for the predicted vs expected values
            BaseMetric metric = new MeanAbsoluteError();
            var metricValues = metric.Calculate(preds, y);
            metricValues.Print("MAE Metric");

            Console.ReadLine();
        }
    }
}
