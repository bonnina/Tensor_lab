using System;
using System.Linq;

namespace Tensor_lab
{
    public class Tensor
    {
        private double[] data;

        public int[] Shape { get; set; }
        public int Elements
        {
            get {
                return Shape.Aggregate((a, b) => a * b);
            }
        }

        public Tensor(params int[] shape)
        {
            Shape = shape;
            data = new double[Elements];
        }

        public void Load(params double[] dataArr)
        {
            data = dataArr;
        }

        public void Fill(double value)
        {
            for(int i = 0; i < Elements; i++)
            {
                data[i] = value;
            }
        }
    }
}
