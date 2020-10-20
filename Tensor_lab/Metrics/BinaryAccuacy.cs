using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.Metrics
{
    public class BinaryAccuacy : BaseMetric
    {
        public BinaryAccuacy() : base("binary_accurary")
        {
        }

        public override Tensor Calculate(Tensor preds, Tensor labels)
        {
            var output = Round(Clip(preds, 0, 1));
            return Mean(preds == labels);
        }
    }
}
