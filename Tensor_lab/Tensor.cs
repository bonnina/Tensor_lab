using System;
using System.Linq;

namespace Tensor_lab
{
    public class Tensor
    {
        private double[] data;

        /// <summary>
        /// Shape of the dataset - a matrix.
        /// </summary>
        public int[] Shape { get; set; }

        /// <summary>
        /// The number of elements the Shape array will hold.
        /// </summary>
        public int Elements
        {
            get {
                return Shape.Aggregate((a, b) => a * b);
            }
        }

        /// <param name="shape"></param>
        public Tensor(params int[] shape)
        {
            Shape = shape;
            data = new double[Elements];
        }

        /// <summary>
        /// Helper function to load the data intos the Tensor
        /// </summary>
        /// <param name="data"></param>
        public void Load(params double[] dataArr)
        {
            data = dataArr;
        }

        /// <summary>
        /// Fill the array with constant value (e.g. weights or bias)
        /// </summary>
        /// <param name="value"></param>
        public void Fill(double value)
        {
            for(int i = 0; i < Elements; i++)
            {
                data[i] = value;
            }
        }

        /// <param name="indices"></param>
        /// <returns></returns>
        public double this[params int[] indices]
        {
            get
            {
                long index = GetIndex();

                return data[index];
            }

            set
            {
                long index = GetIndex();

                data[index] = value;
            }
        }

        /// <returns></returns>
        public int[] GetStride()
        {
            var stride = new int[Shape.Length];
            int acc = 1;

            for (int i = Shape.Length -1; i >= 0; --i)
            {
                stride[i] = acc;
                acc *= Shape[i];
            }

            return stride;
        }

        /// <param name="indices"></param>
        /// <returns></returns>
        public long GetIndex(params int[] indices)
        {
            long index = 0;
            var strides = GetStride();

            for (var i = 0; i < indices.Length; ++i)
            {
                index += indices[i] * strides[i];
            }

            return index;
        }

        /// <summary>
        /// Print the dataset in a matrix form.
        /// </summary>
        public void Print()
        {
            for(int i = 0; i < Shape[0]; i++)
            {
                for(int j = 0; j < Shape[1]; j++)
                {
                    Console.WriteLine(this[i, j]);
                }

                Console.WriteLine();
                Console.WriteLine();
            }
        }

        #region operators

        /// <summary>
        /// Adds two Tensor arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator +(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for(int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] + b[i];
            }

            return t;
        }

        /// <summary>
        /// Subtracts two Tensor arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator -(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] - b[i];
            }

            return t;
        }

        #endregion
    }
}
