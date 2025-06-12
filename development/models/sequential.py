__all__ = [
    "Sequential"
]

import copy
from os import path

from typing import (
    List, Dict, Union, OrderedDict, Iterable, Callable
)
from tqdm.auto import tqdm


import torch
from torch import nn
from torch.utils import data

import matplotlib.pyplot as plt

from ..utilis import (
    get_quantize_scale_zero_point_per_tensor_assy,
    quantize_per_tensor_assy,
    dequantize_per_tensor_assy,
    
    convert_tensor_to_bytes_var,
    int8_to_bytes,

    QUANTIZATION_NONE,
    DYNAMIC_QUANTIZATION_PER_TENSOR,
    DYNAMIC_QUANTIZATION_PER_CHANNEL,
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
)

class Sequential(nn.Sequential):

    def __init__(self, *args):

        if len(args) == 0 and isinstance(args[0], OrderedDict):
            super(Sequential, self).__init__(*args)
        else:
            super(Sequential, self).__init__()
            class_idx = dict()
            for layer in args:
                class_idx[layer.__class__.__name__] = class_idx.get(layer.__class__.__name__, -1) + 1
                idx = class_idx[layer.__class__.__name__]
                layer_type = layer.__class__.__name__.lower()
                self.add_module(f"{layer_type}_{idx}", layer)
                # class_idx[layer.__class__.__name__] += 1

        self.layers = dict()
        self.fit_history = dict()

        for name, layer in self.named_children():
            self.layers[name] = layer

        self.optim_fn = None
        self.criterion_fun = None
        self.lr_scheduler = None

    def set_optimizer(self, optim_fun, *args, **kwargs) -> None:
        self.optim_fn  = optim_fun(self.parameters(), *args, **kwargs)

    def set_criterion(self, criterion_fun, *args, **kwargs) -> None:
        self.criterion_fun = criterion_fun(*args, **kwargs)

    def set_lr_scheduler(self, lr_scheduler, *args, **kwargs) -> None:
        self.lr_scheduler = lr_scheduler(self.optim_fn, *args, **kwargs)



    def forward(self, input):
        setattr(self, "input_shape", input.size())
        if not hasattr(self, "test_input"):
            import random # random.randint(0, input.size(0)-1)
            index = random.randint(0, input.size(0)-1)
            print(f"unsing this {index}")
            test_input = input[index].unsqueeze(dim=0).cpu()

            setattr(self, "test_input", test_input)
            setattr(self, "test_input_quant", quantize_per_tensor_assy(test_input, self.input_scale, self.input_zero_point, self.quantization_bitwidth))

        if hasattr(self, "quantization_type") and getattr(self, "quantization_type") ==  STATIC_QUANTIZATION_PER_TENSOR:
            input = quantize_per_tensor_assy(input, self.input_scale, self.input_zero_point, self.quantization_bitwidth)

        elif hasattr(self, "quantization_type") and getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
            input = quantize_per_tensor_assy(input, self.input_scale, self.input_zero_point, self.quantization_bitwidth)

        for layer in self:
        
            # Only apply layers that are actual torch modules (e.g. nn.Conv2d, nn.ReLU)
            if isinstance(layer, nn.Module) and not isinstance(layer, (type(self.optim_fn), type(self.criterion_fun), type(self.lr_scheduler))):
                input = layer(input)
        return input


    def fit(self, train_dataloader: data.DataLoader, epochs:int, validation_dataloader: data.DataLoader = None, metrics:Union[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]], None]=None, device:str = "cpu") -> Dict[str, List[float]]:
        history = dict()
        metrics_val = dict()

        assert self.optim_fn is not None and self.criterion_fun is not None, "Set your Optimizer and Criterion funtion."

        for epoch in tqdm(range(epochs)):
            train_loss = 0
            if metrics is not None:
                for name in metrics:
                    metrics_val[f"train_{name}"] = 0

            self.train()
            for X, y_true in train_dataloader:
                X = X.to(device)
                y_true = y_true.to(device)

                self.optim_fn.zero_grad()

                y_pred = self(X)
                loss = self.criterion_fun(y_pred, y_true)
                loss.backward()
                train_loss += loss.item()

                if metrics is not None:
                    for name, func in metrics.items():
                        metrics_val[f"train_{name}"] +=  func(y_pred, y_true)
                        # (y_pred.argmax(dim=1) == y_true).sum().item()

                self.optim_fn.step()

            train_loss /= len(train_dataloader.dataset)

            if metrics is not None:
                metrics_val[f"train_{name}"] /= len(train_dataloader.dataset)


            if validation_dataloader is not None:
                self.eval()
                with torch.inference_mode():
                    validation_loss = 0
                    if metrics is not None:
                        for name in metrics:
                            metrics_val[f"validation_{name}"] = 0
                    for X, y_true in validation_dataloader:

                        X = X.to(device)
                        y_true = y_true.to(device)

                        y_pred = self(X)
                        validation_loss += self.criterion_fun(y_pred, y_true).item()
                        if metrics is not None:
                            for name, func in metrics.items():
                                metrics_val[f"validation_{name}"] +=  func(y_pred, y_true)
                        # validation_accuracy += (y_pred.argmax(dim=1) == y_true).sum().item()

                validation_loss /= len(validation_dataloader.dataset)
                if metrics is not None:
                    metrics_val[f"validation_{name}"] /= len(validation_dataloader.dataset)
                    # validation_accuracy /= len(validation_dataloader.dataset)

            if self.lr_scheduler is not None: self.lr_scheduler.step(validation_loss)

            if validation_dataloader is None:

                print(f"epoch {epoch:4d} train loss {train_loss:.4f}", end="")# train accuracy {train_accuracy:.4f}")
                if metrics is not None:
                    for name in metrics:
                        print(f" train {name:.4f} {metrics[f'train_{name}']}", end="")
                print()
            else:
                print(f"epoch {epoch:4d} train loss {train_loss:.4f} validation loss {validation_loss:4f}", end="")# train accuracy {train_accuracy:.4f} validation accuracy {validation_accuracy:.4f}")
                if metrics is not None: 
                    for name in metrics:
                        print(f" validation {name:.4f} {metrics[f'validation_{name}']}", end="")
                print()

                self.fit_history["validation_loss"] = self.fit_history.get("validation_loss", []) + [validation_loss]
                if metrics is not None: 
                    for name in metrics:
                        self.fit_history[f"validation_{name}"] = self.fit_history.get(f"validation_{name}", []) + [metrics[f"validation_{name}"]]
                history["validation_loss"] = history.get("validation_loss", []) + [validation_loss]
                if metrics is not None: 
                    for name in metrics:
                        history[f"validation_{name}"] = history.get(f"validation_{name}", []) + [metrics[f"validation_{name}"]]

                # history["validation_accuracy"] = history.get("validation_accuracy", []) + [validation_accuracy]

            self.fit_history["train_loss"] = self.fit_history.get("train_loss", []) + [train_loss]
            if metrics is not None: 
                    for name in metrics:
                        self.fit_history[f"train_{name}"] = self.fit_history.get(f"train_{name}", []) + [metrics[f"train_{name}"]]

            # self.fit_history["train_accuracy"] = self.fit_history.get("train_accuracy", []) + [train_accuracy]
            
            history["train_loss"] = history.get("train_loss", []) + [train_loss]
            if metrics is not None: 
                for name in metrics:
                    history[f"train_{name}"] = history.get(f"train_{name}", []) + [metrics[f"train_{name}"]]
            # history["train_accuracy"] = history.get("train_accuracy", []) + [train_accuracy]
            
        for layer in self.layers:
            if hasattr(layer, "set_buffer"): layer.set_buffer()

        return history

    @torch.no_grad()
    def revert_param(self):
        for layer in self.layers():
            if hasattr(layer, "reset_buffer"): layer.reset_buffer()
        
        if hasattr(self, "quantization_type"): setattr(self, "quantization_type", QUANTIZATION_NONE)
        return

    @torch.inference_mode()
    def evaluate(self, data_loader: data.DataLoader, device:str = "cpu") -> float:
        self.eval()
        accuracy = 0
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            y_pred = self(X)
            accuracy += (y_pred.argmax(dim=1) == y_true).sum().item()
            #################################
            break
        accuracy /= X.size(0)
            ##############################
        # accuracy /= len(data_loader.dataset)

        return accuracy * 100



    def get_weight_distributions(self, bins=256) -> Dict[str, Union[torch.Tensor, None]]:
        weight_dist = dict()
        for name, layer in self.layers.items():
            if hasattr(layer, "weight"): weight_dist[name] = torch.histogram(layer.weight.cpu(), bins=bins)
            else: weight_dist[name] = None

        return weight_dist


    @torch.no_grad()
    def get_layers_prune_channel_sensity(self, data_loader:data.DataLoader, sparsities:Dict[str, List], metric:str = "l2", device="cpu") -> Dict[str, List[float]]:
        history = dict()

        default_config = dict()
        for name in self.layers.keys():
            default_config[name] = 0.2

        for name in self.layers.keys():
            history[name] = []

            for sparsity in tqdm(sparsities[name], desc=f"Pruning {name}"):
                config = default_config.copy()
                config[name] = sparsity
                model = self.prune_channel(config, metric=metric)
                history[name] += [{sparsity : model.evaluate(data_loader, device=device)}]
                print(history)

        for layer, records in history.items():
            sparsities = []
            scores = []
            for record in records:
                for s, score in record.items():
                    sparsities.append(s)
                    scores.append(score)
            plt.plot(sparsities, scores, label=layer, marker='o')

        plt.xlabel("Sparsity")
        plt.ylabel("Evaluation Score")
        plt.title("Layer-wise Pruning Sensitivity")
        plt.grid(True)
        plt.legend()
        plt.show()

        return history


    def prune_channel(self, sparsity:Union[Dict[str, float], float], metric:str = "l2") -> None:

        model = copy.deepcopy(self)

        keep_prev_channel_index = None

        if isinstance(sparsity, float):
            sparsity_temp = dict()
            for name, layer in model.layers.items():
                sparsity_temp[name] = sparsity
            sparsity = sparsity_temp
            
        for name, layer in list(model.layers.items())[:-1]:
            if hasattr(layer, "prune_channel"):
                keep_prev_channel_index = layer.prune_channel(sparsity[name], keep_prev_channel_index, is_output_layer=False, metric=metric)

        name, layer = list(model.layers.items())[-1]
        if hasattr(layer, "prune_channel"):
            keep_prev_channel_index = layer.prune_channel(sparsity[name], keep_prev_channel_index, is_output_layer=True, metric=metric)
        return model


    @torch.no_grad()
    def get_dynamic_quantize_per_tensor_sensity(self, data_loader:data.DataLoader, bitwidths:Iterable[float], device:str = "cpu") -> Dict[float, List[float]]:
        history = dict()

        for bitwidth in bitwidths:
            model = self.dynamic_quantize_per_tensor(bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history

    @torch.no_grad()
    def dynamic_quantize_per_tensor(self, bitwidth:int = 8):
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_bitwidth", bitwidth)
        setattr(model, "quantization_type", DYNAMIC_QUANTIZATION_PER_TENSOR)

        for layer in model.layers.values():
            if hasattr(layer, "dynamic_quantize_per_tensor"):
                layer.dynamic_quantize_per_tensor(bitwidth)

        return model



    @torch.no_grad()
    def get_dynamic_quantize_per_channel_sensity(self, data_loader:data.DataLoader, bitwidths:Iterable[float], device:str = "cpu") -> Dict[float, List[float]]:
        history = dict()

        for bitwidth in bitwidths:
            model = self.dynamic_quantize_per_channel(bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history

    @torch.no_grad()
    def dynamic_quantize_per_channel(self, bitwidth:int = 8):
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", DYNAMIC_QUANTIZATION_PER_CHANNEL)
        setattr(model, "quantization_bitwidth", bitwidth)


        for layer in model.layers.values():
            if hasattr(layer, "dynamic_quantize_per_channel"):
                layer.dynamic_quantize_per_channel(bitwidth)

        return model




    @torch.no_grad()
    def get_static_quantize_per_tensor_sensity(self, input_batch_real:torch.Tensor, data_loader:data.DataLoader, bitwidths:Iterable[float], device:str = "cpu") -> Dict[float, List[float]]:
        history = dict()

        for bitwidth in bitwidths:
            model = self.static_quantize_per_tensor(input_batch_real, bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history


    @torch.no_grad()
    def static_quantize_per_tensor(self, input_batch_real:torch.Tensor, bitwidth:int = 8):
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)
        setattr(model, "quantization_bitwidth", bitwidth)

        input_scale, input_zero_point = get_quantize_scale_zero_point_per_tensor_assy(input_batch_real, bitwidth)
        input_batch_quant = quantize_per_tensor_assy(input_batch_real, input_scale, input_zero_point, bitwidth)
        # input_batch_quant, input_scale, input_zero_point, input_quant_dequant = quantize_per_tensor_assy(input_batch_real, bitwidth)

        model.register_buffer("input_scale", input_scale)
        model.register_buffer("input_zero_point", input_zero_point)

        for layer in model.layers.values():
            if hasattr(layer, "static_quantize_per_tensor"):

                input_batch_real, input_batch_quant, \
                input_scale, input_zero_point = layer.static_quantize_per_tensor(
                                                    input_batch_real, input_batch_quant,
                                                    input_scale, input_zero_point,
                                                    bitwidth,
                                                )
            else:
                raise AttributeError(f"Static Quantization Per Tensor not implement for {layer.__class__.__name__}!!!")


        model.register_buffer("output_scale", input_scale)
        model.register_buffer("output_zero_point", input_zero_point)

        return model




    @torch.no_grad()
    def get_static_quantize_per_channel_sensity(self, input_batch_real:torch.Tensor, data_loader:data.DataLoader, bitwidths:Iterable[float], device:str = "cpu") -> Dict[float, List[float]]:
        history = dict()

        for bitwidth in bitwidths:
            model = self.static_quantize_per_channel(input_batch_real, bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history


    @torch.no_grad()
    def static_quantize_per_channel(self, input_batch_real:torch.Tensor, bitwidth:int = 8):
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)
        setattr(model, "quantization_bitwidth", bitwidth)

        input_channel_axis = 1

        input_scale, input_zero_point = get_quantize_scale_zero_point_per_tensor_assy(input_batch_real, bitwidth)
        input_batch_quant = quantize_per_tensor_assy(input_batch_real, input_scale, input_zero_point, bitwidth)
        # input_batch_quant, input_scale, input_zero_point, input_quant_dequant = quantize_per_tensor_assy(input_batch_real, bitwidth)

        model.register_buffer("input_scale", input_scale)
        model.register_buffer("input_zero_point", input_zero_point)

        for layer in model.layers.values():
            if hasattr(layer, "static_quantize_per_channel"):

                input_batch_real, input_batch_quant, \
                input_scale, input_zero_point = layer.static_quantize_per_channel(
                                                    input_batch_real, input_batch_quant,
                                                    input_scale, input_zero_point,
                                                    bitwidth,
                                                )
            else:
                raise AttributeError(f"Static Quantization Per Tensor not implement for {layer.__class__.__name__}!!!")


        model.register_buffer("output_scale", input_scale)
        model.register_buffer("output_zero_point", input_zero_point)

        return model

    def get_max_workspace_arena(self):


        # Create random input tensor based on model's expected input shape, it is saved as a batched tensor
        input_shape = list(self.input_shape)
        input_shape[0] = 1 
        x = torch.randn(input_shape)

        max_output_even_size = x.numel()
        max_output_odd_size = 0
        
        for i, layer in enumerate(self.layers.values(), start=1):

            x = layer(x)

            if (i%2 == 0):
                max_output_even_size = max(max_output_even_size, x.numel())  # Track max number of elements
            else:
                max_output_odd_size = max(max_output_odd_size, x.numel())  # Track max number of elements

        return max_output_even_size, max_output_odd_size



    @torch.no_grad()
    def convert_to_c(self, var_name, dir="./"):

        def write_str_to_c_file(file_str:str, file_name:str, dir:str):
            with open(path.join(dir, file_name), "w") as file:
                file.write(file_str)
        
        header_file = f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n"
        header_file += "#include <stdint.h>\n\n\n"

        definition_file = f"#include \"{var_name}.h\"\n\n"
        param_definition_file = f"#include \"{var_name}.h\"\n\n"
    
        max_output_even_size, max_output_odd_size = self.get_max_workspace_arena()

        if getattr(self, "quantization_type", QUANTIZATION_NONE) != STATIC_QUANTIZATION_PER_TENSOR:

            workspace_header = (
                f"#define MAX_OUTPUT_EVEN_SIZE {max_output_even_size}\n"
                f"#define MAX_OUTPUT_ODD_SIZE {max_output_odd_size}\n"
                f"extern float workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )

            workspace_def = (
                f"float workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )

        else:
            workspace_header = (
                f"#define MAX_OUTPUT_EVEN_SIZE {max_output_even_size}\n"
                f"#define MAX_OUTPUT_ODD_SIZE {max_output_odd_size}\n"
                f"extern int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )

            workspace_def = (
                f"int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )

        header_file += workspace_header
        definition_file += workspace_def

        
        layers_header = (
            f"#define LAYERS_LEN {len(self.layers)}\n"
            f"extern Layer* layers[LAYERS_LEN];\n\n"
            f"extern Sequential {var_name};\n\n"
        )

        layers_def = (
            f"{self.__class__.__name__} {var_name}(layers, LAYERS_LEN, workspace, MAX_OUTPUT_EVEN_SIZE);\n"
            f"\nLayer* layers[LAYERS_LEN] = {{\n"
        )
        for layer_name, layer in self.layers.items():
            if hasattr(layer, "convert_to_c"):
                layers_def += f"    &{layer_name},\n"
                layer_header, layer_def, layer_param_def = layer.convert_to_c(layer_name)
                layers_header += layer_header
                param_definition_file += layer_param_def
                definition_file += layer_def 
            else:
                raise RuntimeError(f"The cpp conversion of {layer.__class__.__name__} has not been implemented!")
        
        layers_def += "};\n"
        definition_file += layers_def

        header_file += layers_header

        header_file += f"\n#endif //{var_name.upper()}_h\n"

        write_str_to_c_file(header_file, f"{var_name}.h", dir)
        write_str_to_c_file(definition_file, f"{var_name}_def.cpp", dir)
        write_str_to_c_file(param_definition_file, f"{var_name}_params.cpp", dir)

        # _, input_def = "", ""
        _, test_input_def = convert_tensor_to_bytes_var(self.test_input, "test_input", self.quantization_bitwidth)
        # _, test_input_def1 = convert_tensor_to_bytes_var(self.test_input_quant, "test_input_quant", self.quantization_bitwidth)
        # input_def = test_input_def0 + test_input_def1

        var_def_str = f"\nconst uint8_t test_input_quant[] = {{\n"

        for line in torch.split(self.test_input_quant.flatten(), 8):
            var_def_str += "    " + ", ".join(
                [f"0x{b:02X}" for val in line for b in int8_to_bytes(val)]
        ) + ",\n"

        var_def_str += "};\n"

        with open(path.join(dir, f"{var_name}_test_input.h"), "w") as file:
            file.write(test_input_def+var_def_str)

        return