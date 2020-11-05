
namespace Tensor_lab.Layers.Activations
{
    public class Sigmoid : BaseActivation
    {
        public Sigmoid() : base("sigmoid") {}

        public override void Forward(Tensor x)
        {
            base.Forward(x);
            Output = GetExp(x) / (1 + GetExp(x));
        }

        public override void Backward(Tensor grad)
        {
            InputGrad = grad * Output * (1 - Output);
        }
    }
}
