using System;
using System.Collections.Generic;
using Tensor_lab.Layers.Activations;

namespace Tensor_lab.Layers
{
    public class FullyConnectedLayer : BaseLayer
    {
        public int InputDim { get; set; }
        public int OutNeurons { get; set; }
        public BaseActivation Activation { get; set; }

        public FullyConnectedLayer(int input_dim, int output_neurons, string act) : base("fc")
        {
            Parameters["w"] = GetRandom(input_dim, output_neurons);
            InputDim = input_dim;
            OutNeurons = output_neurons;
            Activation = BaseActivation.Get(act);
        }
        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = GetDotProduct(x, Parameters["w"]);
            if (Activation != null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }
    }
}
