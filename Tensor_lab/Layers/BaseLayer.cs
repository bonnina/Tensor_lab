using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.Layers
{
    public abstract class BaseLayer : Operations
    {
        public string Name { get; set; }

        public Tensor Input { get; set; }

        public Tensor Output { get; set; }

        public Dictionary<string, Tensor> Parameters { get; set; }

        public BaseLayer(string name)
        {
            Name = name;
            Parameters = new Dictionary<string, Tensor>();
        }
        
        public virtual void Forward(Tensor t)
        {
            Input = t;
        }
    }
}
