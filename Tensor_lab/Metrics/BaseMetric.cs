using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.Metrics
{
    public abstract class BaseMetric : Operations
    {
        public string Name { get; set; }

        public BaseMetric(string name)
        {
            Name = name;
        }

        public abstract Tensor Calculate(Tensor preds, Tensor labels);
    }
}
