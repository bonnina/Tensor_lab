using System.Collections.Generic;

namespace Tensor_lab.Layers
{
    public abstract class BaseLayer : Operations
    {
        /// <summary>
        /// Layer name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Input for the layer
        /// </summary>
        public Tensor Input { get; set; }

        /// <summary>
        /// Output after forwarding the input across the neurons
        /// </summary>
        public Tensor Output { get; set; }

        /// <summary>
        /// Trainable parameters list, eg, weight, bias
        /// </summary>
        public Dictionary<string, Tensor> Parameters { get; set; }

        /// <summary>
        /// Gradient of the Input
        /// </summary>
        public Tensor InputGrad { get; set; }

        /// <summary>
        /// List of all parameters gradients calculated during back propagation.
        /// </summary>
        public Dictionary<string, Tensor> Grads { get; set; }

        public BaseLayer(string name)
        {
            Name = name;
            Parameters = new Dictionary<string, Tensor>();
        }
        
        /// <summary>
        /// Virtual forward method to perform calculation and move the input to next layer
        /// </summary>
        /// <param name="x"></param>
        public virtual void Forward(Tensor t)
        {
            Input = t;
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public virtual void Backward(Tensor grad) {}

        public void PrintParams(bool printGrads = true)
        {
            foreach (var item in Parameters)
            {
                item.Value.Print(string.Format("Parameter: {0}", item.Key));
                if (printGrads && Grads.ContainsKey(item.Key))
                {
                    Grads[item.Key].Print(string.Format("Grad: {0}", item.Key));
                }
            }
        }
    }
}
