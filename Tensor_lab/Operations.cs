using System;
using System.Collections.Generic;
using System.Text;

namespace Tensor_lab
{
    public class Operations
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
        /// Calculates square root of the tensor
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public Tensor GetSqrt(Tensor t)
        {
            Tensor result = new Tensor(t.Shape);

            for (int i = 0; i < t.Elements; i++)
            {
                result[i] = Math.Sqrt(t[i]);
            }

            return result;
        }

        /// <summary>
        /// Calculates the square of the tensor
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public Tensor GetPow(Tensor t)
        {
            Tensor result = new Tensor(t.Shape);

            for (int i = 0; i < t.Elements; i++)
            {
                result[i] = Math.Pow(t[i], 2);
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

        /// <summary>
        /// Transpose the matrix by turning all the rows into columns and vice-versa.
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public Tensor Pivot(Tensor t)
        {
            Tensor result = new Tensor(t.Shape);
            for (int i = 0; i < t.Shape[0]; i++)
            {
                for (int j = 0; j < t.Shape[1]; j++)
                {
                    result[i, j] = t[j, i];
                }
            }
            return result;
        }

        /// <summary>
        /// Clip the values in tensor between min and max values
        /// </summary>
        /// <param name="x"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public Tensor Clip(Tensor x, double min, double max)
        {
            Tensor result = new Tensor(x.Shape);
            for (int i = 0; i < x.Elements; i++)
            {
                result[i] = (x[i] < min) ? min : (x[i] > max) ? max : x[i];
            }

            return result;
        }

        /// <summary>
        /// Round the values in the tensor
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public Tensor Round(Tensor x)
        {
            Tensor result = new Tensor(x.Shape);
            for (int i = 0; i < x.Elements; i++)
            {
                result[i] = Math.Round(x[i]);
            }

            return result;
        }
    }
}
