import torch

class LoRALinear(torch.nn.Module):
    def __init__(self, in_params: int, out_params: int, r: int):
        super(LoRALinear, self).__init__()
        self.alpha = 1.0
        self.main = torch.nn.Linear(in_params, out_params, bias=False)
        self.lora_down = torch.nn.Linear(out_params, r, bias=False)
        self.lora_up = torch.nn.Linear(r, out_params, bias=False)

        torch.nn.init.normal_(self.lora_down.weight, std=1/16)
        torch.nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        x = self.main(x) + self.alpha * self.lora_up(self.lora_down(x))
        return x

def replace_with_LoRALinear(model: torch.nn.Module, layer_name: str, r: int):
    old_layer = model._modules[layer_name]
    new_layer = LoRALinear(old_layer.in_features, old_layer.out_features, r)
    new_layer.main.weight.data = old_layer.weight.data
    if old_layer.bias is not None:
        new_layer.main.bias.data = old_layer.bias.data
    else:
        new_layer.main.bias = None
    model._modules[layer_name] = new_layer

def replace_all_matching_layers(model: torch.nn.Module, r: int, matches: list):
    for name, layer in model._modules.items():
        if True in [match in name for match in matches]:
            replace_with_LoRALinear(model, name, r)

def save_lora_layers(model: torch.nn.Module, filename: str):
    lora_layers = {}
    for name, layer in model._modules.items():
        if isinstance(layer, LoRALinear):
            lora_layers[name] = {
                "lora_down": layer.lora_down.state_dict(),
                "lora_up": layer.lora_up.state_dict()
            }
    torch.save(lora_layers, filename)

def load_lora_layers(model: torch.nn.Module, filename: str):
    lora_layers = torch.load(filename)
    for name, layer in model._modules.items():
        if isinstance(layer, LoRALinear) and name in lora_layers:
            layer.lora_down.load_state_dict(lora_layers[name]["lora_down"])
            layer.lora_up.load_state_dict(lora_layers[name]["lora_up"])

def change_alpha_parameter(model: torch.nn.Module, alpha: float):
    for name, layer in model._modules.items():
        if isinstance(layer, LoRALinear):
            layer.alpha = alpha
