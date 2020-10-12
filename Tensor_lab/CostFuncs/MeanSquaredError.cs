using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab.CostFuncs
{
    public class MeanSquaredError : BaseCost
    {
        public MeanSquaredError() : base("mean_squared_error")
        {

        }
        public override Tensor Forward(Tensor preds, Tensor labels)
        {
            var error = preds - labels;
            return Mean(GetPow(error));
        }
    }
}
