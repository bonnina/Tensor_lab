using System;
using System.Linq;

namespace Tensor_lab
{
    public class Tensor
    {
        private double[] data;

        /// <summary>
        /// Shape of the dataset - a matrix, can be from 1D to 3D.
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
    }
}
