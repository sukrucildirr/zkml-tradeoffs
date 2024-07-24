import ezkl
import json
import torch
from train_model import SimplePerceptron

def setup(model, onnx_file, input_sample):
    INPUT = "input_data.json"
    SETTINGS = "settings.json"
    CALIBRATION = "calibration.json"
    WITNESS = "witness.json"
    COMPILED_MODEL = "compiled_model.json"
    VK = "vk.json"
    PK = "pk.json"
    PROOF = "proof.pf"

    input_data = {
        'input_shapes': list(input_sample.shape),
        'input_data': input_sample.detach().numpy().tolist(),
        "output_data": model(input_sample).detach().numpy().tolist()
    }

    json.dump(input_data, open(INPUT, 'w'))

    assert ezkl.gen_settings(onnx_file, SETTINGS)
    json.dump(input_data, open(CALIBRATION, 'w'))
    assert ezkl.calibrate_settings(INPUT, onnx_file, SETTINGS, "resources")
    assert ezkl.compile_circuit(onnx_file, COMPILED_MODEL, SETTINGS)
    assert ezkl.get_srs(SETTINGS)
    ezkl.gen_witness(INPUT, COMPILED_MODEL, WITNESS)
    assert ezkl.setup(COMPILED_MODEL, VK, PK)

def main():
    input_size = 28 * 28
    output_size = 2
    model = SimplePerceptron(input_size, output_size)
    model.load_state_dict(torch.load("models/perceptron.pth"))
    model.eval()

    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, "models/perceptron.onnx", input_names=['input'], output_names=['output'])
    setup(model, "models/perceptron.onnx", dummy_input)

if __name__ == "__main__":
    main()