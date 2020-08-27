using System;
using Tensor_lab.Layers;

namespace Tensor_lab.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            //Load array to the tensor
            Tensor x = new Tensor(1, 3);
            x.Load(1, 2, 3);
            x.Print("Load array");

            //Create two layers, one with 6 neurons and another with 1
            FullyConnectedLayer fc1 = new FullyConnectedLayer(3, 6, "relu");
            FullyConnectedLayer fc2 = new FullyConnectedLayer(6, 1, "sigmoid");

            //Connect input by passing data from one layer to another
            fc1.Forward(x);
            x = fc1.Output;
            x.Print("FC1 Output");
            fc2.Forward(x);
            x = fc2.Output;
            x.Print("FC2 Output");

            Console.ReadLine();
        }
    }
}
