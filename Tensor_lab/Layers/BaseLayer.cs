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
    }
}
