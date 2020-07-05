using System;

namespace Tensor_lab.Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Operations Check1 = new Operations();

            // Load array to the tensor
            Tensor a = new Tensor(3, 6);
            a.Load(1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            a.Print();

            // Transpose the matrix
            Tensor t = a.Transpose();
            t.Print();

            // Create a tensor with all values of 5
            Tensor b = new Tensor(6, 3);
            b.Fill(5);
            b.Print();

            // Create a tensor with all values of 3
            Tensor c = new Tensor(6, 3);
            c.Fill(3);
            c.Print();

            // Subtract two tensors
            b = b - c;

            // Perform the dot product
            Tensor r = Check1.GetDotProduct(a, b);
            r.Print();

            Console.ReadLine();
        }
    }
}
