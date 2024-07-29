import torch


from safetensors import safe_open
from safetensors.torch import save_file


def torch_dict_load(file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {}
            for k in f.keys():
                d[k] = f.get_tensor(k)
        return d
    raise ValueError("File extension not supported")

        
def torch_dict_save(data, file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")


def fusion_pretraining_load(model, weights):
    incompatible_keys = model.load_state_dict(weights, strict=False)
    expected_unexpected_keys = ["model.model.0.conv.weight"]
    expected_missing_keys = [
        "model.model.0.optional_rgb.conv.weight",
        "model.model.0.optional_rgb.conv.bias",
        "model.model.0.optional_rgb.bn.weight",
        "model.model.0.optional_rgb.bn.bias",
        "model.model.0.optional_rgb.bn.running_mean",
        "model.model.0.optional_rgb.bn.running_var",
        "model.model.0.optional_rgb.embedding.weight",
        "model.model.0.optional_ir.conv.weight",
        "model.model.0.optional_ir.conv.bias",
        "model.model.0.optional_ir.bn.weight",
        "model.model.0.optional_ir.bn.bias",
        "model.model.0.optional_ir.bn.running_mean",
        "model.model.0.optional_ir.bn.running_var",
        "model.model.0.optional_ir.embedding.weight",
        "model.model.0.conv.conv.weight",
        "model.model.0.conv.bn.weight",
        "model.model.0.conv.bn.bias",
        "model.model.0.conv.bn.running_mean",
        "model.model.0.conv.bn.running_var",
    ]
    assert (
        incompatible_keys.unexpected_keys == expected_unexpected_keys
    ), f"Expected {expected_unexpected_keys}, got {incompatible_keys.unexpected_keys}"
    assert (incompatible_keys.missing_keys == expected_missing_keys), f"Expected {expected_missing_keys}, got {incompatible_keys.missing_keys}"


def nc_safe_load(model, weights, nc):
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as e:
        error_msg = str(e)
        pattern = r"size mismatch for ([\w.]+): copying a param with shape torch.Size\((\[.*?\])\) from checkpoint, the shape in current model is torch.Size\((\[.*?\])\)"
        matches = re.findall(pattern, error_msg)
        for m in matches:
            print(f"Detected mismatch in {m[0]}: {m[1]} vs {m[2]}")
        checkpoint_shapes = [x for m in matches for x in eval(m[1])]
        model_shapes = [x for m in matches for x in eval(m[2])]
        diffs = [
            diff for diff in zip(checkpoint_shapes, model_shapes) if diff[0] != diff[1]
        ]
        for diff in diffs:
            assert (
                diff[1] == nc
            ), "Detected a mismatch which is not due to the number of classes"
        print(f"Loading model with {nc} classes")
