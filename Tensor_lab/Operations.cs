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

        /// <summary>
        /// Calculates the exponential of the tensor
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public Tensor GetExp(Tensor t)
        {
            Tensor result = new Tensor(t.Shape);

            for (int i = 0; i < t.Elements; i++)
            {
                result[i] = Math.Exp(t[i]);
            }

            return result;
        }

        /// <summary>
        /// Calculates the logrithmic of the tensor
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public Tensor GetLog(Tensor t)
        {
            Tensor result = new Tensor(t.Shape);

            for (int i = 0; i < t.Elements; i++)
            {
                result[i] = Math.Log(t[i]);
            }

            return result;
        }

        /// <summary>
        /// Perform the dot product between two matrices. Only applicable to 2D Tensor. 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public Tensor GetDotProduct(Tensor a, Tensor b)
        {
            if (a.Shape[1] != b.Shape[0])
            {
                throw new Exception("a->Cols must be equal to b->Rows");
            }
            int m = a.Shape[0];
            int q = b.Shape[1];
            int n = a.Shape[1];
            Tensor r = new Tensor(m, q);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < q; j++)
                {
                    r[i, j] = 0;
                    for (int k = 0; k < n; k++)
                    {
                        r[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return r;
        }
    }
}
