
namespace Tensor_lab.Layers.Activations
{
    public class ReLU : BaseActivation
    {
        public ReLU() : base("relu") { }

        public override void Forward(Tensor x)
        {
            base.Forward(x);

            Tensor matches = x > 0;
            Output = matches * x;
        }

        public override void Backward(Tensor grad)
        {
            InputGrad = grad * (Input > 0);
        }
    }
}
