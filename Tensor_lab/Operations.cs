using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab
{
    class Operations
    {
        /// <summary>
        /// Creates a tensor with random data (declaring weights).
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor GetRandom(params int[] shape)
        {
            var tensor = new Tensor();
            var random = new Random();

            for (int i = 0; i < tensor.Elements; i++) {
                tensor[i] = random.NextDouble();
            }

            return tensor;
        }
    }
}
