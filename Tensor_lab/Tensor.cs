﻿using System;
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

        public double[] Data
        {
            get
            {
                return data;
            }
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

        #region indexer

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

        #endregion

        /// <summary>
        /// Print the dataset in a matrix form.
        /// </summary>
        public void Print(string title = "")
        {
            if (!string.IsNullOrWhiteSpace(title))
            {
                Console.WriteLine(title);
            }

            Console.WriteLine("---------{0}----------", string.Join(" x ", Shape));
            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    Console.Write(Math.Round(this[i, j], 2) + "  ");
                }

                Console.WriteLine();
            }

            Console.WriteLine("-----------------------\n\n");
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

        public static Tensor operator +(Tensor a, double b)
        {
            Tensor t_b = new Tensor(a.Shape);
            t_b.Fill(b);
            return a + t_b;
        }

        public static Tensor operator +(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a + b;
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

        public static Tensor operator -(Tensor a, double b)
        {
            Tensor t_b = new Tensor(a.Shape);
            t_b.Fill(b);
            return a - t_b;
        }

        public static Tensor operator -(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a - b;
        }

        /// <summary>
        /// Negates the values in the tensor
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Tensor operator -(Tensor a)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = -a[i];
            }

            return t;
        }



        /// <summary>
        /// Multiplies two Tensor arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator *(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] * b[i];
            }

            return t;
        }

        public static Tensor operator *(Tensor a, double b)
        {
            Tensor t_b = new Tensor(a.Shape);
            t_b.Fill(b);
            return a * t_b;
        }

        public static Tensor operator *(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a * b;
        }

        /// <summary>
        /// Divides two Tensor arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator /(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] / b[i];
            }

            return t;
        }

        /// <summary>
        /// Check a == b between corresponding Tensor elements
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator ==(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] == b[i] ? 1 : 0;
            }

            return t;
        }

        public static Tensor operator ==(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a == b;
        }

        /// <summary>
        /// Check a != b between corresponding Tensor elements
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator !=(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] != b[i] ? 1 : 0;
            }

            return t;
        }

        public static Tensor operator !=(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a != b;
        }

        /// <summary>
        /// Check a > b between corresponding Tensor elements
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator >(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] > b[i] ? 1 : 0;
            }

            return t;
        }

        public static Tensor operator >(Tensor a, double b)
        {
            Tensor t_b = new Tensor(a.Shape);
            t_b.Fill(b);
            return a > t_b;
        }

        public static Tensor operator >(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a > b;
        }

        /// <summary>
        /// Check a < b between corresponding Tensor elements
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor operator <(Tensor a, Tensor b)
        {
            Tensor t = new Tensor(a.Shape);

            for (int i = 0; i < a.Elements; i++)
            {
                t[i] = a[i] < b[i] ? 1 : 0;
            }

            return t;
        }

        public static Tensor operator <(Tensor a, double b)
        {
            Tensor t_b = new Tensor(a.Shape);
            t_b.Fill(b);
            return a < t_b;
        }

        public static Tensor operator <(double a, Tensor b)
        {
            Tensor t_a = new Tensor(b.Shape);
            t_a.Fill(a);
            return t_a < b;
        }


        #endregion

        /// <summary>
        /// Transposes the axis of 2D array
        /// </summary>
        /// <returns></returns>
        public Tensor Transpose()
        {
            Operations operations = new Operations();
            return operations.Pivot(this);
        }
    }
}
