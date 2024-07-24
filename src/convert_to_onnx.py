import torch
from train_model import SimplePerceptron, load_data

def convert_to_onnx():
    input_size = 28 * 28
    output_size = 2
    model = SimplePerceptron(input_size, output_size)
    model.load_state_dict(torch.load("models/perceptron.pth"))
    model.eval()

    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, "models/perceptron.onnx", input_names=['input'], output_names=['output'])

if __name__ == "__main__":
    convert_to_onnx()