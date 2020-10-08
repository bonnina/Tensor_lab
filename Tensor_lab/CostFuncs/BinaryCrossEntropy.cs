using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.CostFuncs
{
    public class BinaryCrossEntropy : BaseCost
    {
        public BinaryCrossEntropy() : base("binary_crossentropy"){}
        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            var output = Clip(preds, Epsilon, 1 - Epsilon);
            output = Mean(-(labels * GetLog(output) + (1 - labels) * GetLog(1 - output)));
            return output;
        }
    }
}
