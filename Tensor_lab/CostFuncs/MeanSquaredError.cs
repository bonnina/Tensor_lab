﻿using System;
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

        public override Tensor Backward(Tensor preds, Tensor labels)
        {
            double norm = 2 / preds.Shape[0];
            return norm * (preds - labels);
        }
    }
}
