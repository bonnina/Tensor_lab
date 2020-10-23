using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.Metrics
{
    public class MeanAbsoluteError : BaseMetric
    {
        public MeanAbsoluteError() : base("mean_absolute_error")
        {

        }

        public override Tensor Calculate(Tensor preds, Tensor labels)
        {
            var error = preds - labels;
            return Mean(Abs(error));
        }
    }
}
