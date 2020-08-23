using Tensor_lab.Layers.Activations;

namespace Tensor_lab.Layers
{
    /// <summary>
    /// Connects every neuron in one layer to every neuron in another layer. Every neuron does a basic linear operation  of Output = Weights * Input.
    /// </summary>
    public class FullyConnectedLayer : BaseLayer
    {
        /// <summary>
        /// Number of incoming input features (e.g. column length of the matrix)
        /// </summary>
        public int InputDim { get; set; }

        /// <summary>
        /// Number of neurons for the layer
        /// </summary>
        public int OutNeurons { get; set; }

        /// <summary>
        /// Nonlinear activation function for the layer
        /// </summary>
        public BaseActivation Activation { get; set; }

        /// <param name="in">Number of incoming input features</param>
        /// <param name="out">Number of neurons for this layer</param>
        public FullyConnectedLayer(int input_dim, int output_neurons, string act) : base("fc")
        {
            Parameters["w"] = GetRandom(input_dim, output_neurons);
            InputDim = input_dim;
            OutNeurons = output_neurons;
            Activation = BaseActivation.Get(act);
        }

        /// <summary>
        /// Forward the input data by performing calculation across all the neurons, store it in the Output to be accessible by next layer.
        /// </summary>
        /// <param name="x"></param>
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
