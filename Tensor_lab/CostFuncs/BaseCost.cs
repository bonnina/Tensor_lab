using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.CostFuncs
{
    public abstract class BaseCost : Operations
    {
        public string Name { get; set; }

        public BaseCost(string name)
        {
            Name = name;
        }

        public abstract Tensor Forward(Tensor preds, Tensor labels);
    }
}
