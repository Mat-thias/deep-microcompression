"""
@file sequential.py
@brief Extended nn.Sequential container with quantization, pruning and deployment capabilities
"""

__all__ = [
    "Sequential"
]

import copy
from os import path

from typing import (
    List, Dict, OrderedDict, Iterable, Callable, Optional, Union, Any
)
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils import data

# from .callback import EarlyStopper
from ..layers.layer import Layer
from ..utils import (
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
    """Extended Sequential container with additional functionality for:
        - Quantization (dynamic/static, per-tensor/per-channel)
        - Pruning
        - Training utilities
        - C code generation
    """

    def __init__(self, *args):
        """Initialize Sequential model with automatic layer naming
        
        Args:
            *args: Can be either:
                - An OrderedDict of layers
                - Individual layer instances
        """
        setattr(self, "_dmc", dict())

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            super(Sequential, self).__init__(*args)
        else:
            super(Sequential, self).__init__()
            class_idx = dict()
            
            # Auto-name layers with type_index convention (e.g. conv2d_0)
            for layer in args:
                if isinstance(layer, Layer) and isinstance(layer, nn.Module): 
                    class_idx[layer.__class__.__name__] = class_idx.get(layer.__class__.__name__, -1) + 1
                    idx = class_idx[layer.__class__.__name__]
                    layer_type = layer.__class__.__name__.lower()
                    self.add_module(f"{layer_type}_{idx}", layer)
                else:
                    raise TypeError(f"layer of type {type(layer)} isn't a Layer or Module.")

        # Store layers in dict for easy access
        self.layers = dict()
        self.fit_history = dict()
        for name, layer in self.named_children():
            self.layers[name] = layer


    def forward(self, input):
        """Forward pass with quantization support
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor after passing through all layers
        """
################################################################################################
        # Saving the a test input data
        import random
        index = random.randint(0, input.size(0)-1)
        # index = 0
        # print(f"Using this {index}")

        test_input = input[index].unsqueeze(dim=0).cpu()
        # if not hasattr(self, "test_input"):
        setattr(self, "test_input", test_input)
################################################################################################

        # Pass through all layers
        for i, layer in enumerate(self):
            # print(f"Layer {i} ({layer.__class__.__name__}): input shape = {input.shape} kernel_size {getattr(layer, "kernel_size", "nan")} stride {getattr(layer, "stride", "nan")} padding {getattr(layer, "padding", "nan")}")
            input = layer(input)
                
        return input


    def fit(
        self, train_dataloader: data.DataLoader, epochs: int, 
        criterion_fun: torch.nn.Module, 
        optimizer_fun: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        validation_dataloader: Optional[data.DataLoader] = None, 
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
        verbose: bool = True,
        compression_config: Optional[Dict[str, Dict[str, Any]]] = None,
        callbacks: List[Callable] = [],
        device: str = "cpu"
    ) -> Dict[str, List[float]]:
        """Training loop with optional validation and metrics tracking
        
        Args:
            train_dataloader: Training data loader
            epochs: Number of training epochs
            validation_dataloader: Optional validation data loader
            metrics: Dictionary of metric functions {name: function}
            device: Device to run on ('cpu' or 'cuda')
            
        Returns:
            Dictionary of training/validation metrics over time
        """
        history = dict()
        metrics_val = dict()

        for epoch in tqdm(range(epochs)):
            # Training phase
            train_loss = 0
            train_data_len = 0

            if metrics is not None:
                for name in metrics:
                    metrics_val[f"train_{name}"] = 0

            self.train()
            for X, y_true in train_dataloader:
                X = X.to(device)
                y_true = y_true.to(device)

                optimizer_fun.zero_grad()
                y_pred = self(X)
                loss = criterion_fun(y_pred, y_true)
                loss.backward()
                train_loss += loss.item()
                train_data_len += X.size(0)

                if metrics is not None:
                    for name, func in metrics.items():
                        metrics_val[f"train_{name}"] += func(y_pred, y_true)


                if compression_config:
                    self.apply_compression()

                optimizer_fun.step()

# #################################################
#                 break
# #################################################

            train_loss /= train_data_len
            if metrics is not None:
                for name in metrics:
                    metrics_val[f"train_{name}"] /= train_data_len

            # Validation phase
            if validation_dataloader is not None:
                self.eval()
                with torch.inference_mode():
                    validation_loss = 0
                    validation_data_len = 0
                    
                    if metrics is not None:
                        for name in metrics:
                            metrics_val[f"validation_{name}"] = 0
                            
                    for X, y_true in validation_dataloader:
                        X = X.to(device)
                        y_true = y_true.to(device)
                        y_pred = self(X)
                        validation_loss += criterion_fun(y_pred, y_true).item()
                        validation_data_len += X.size(0)
                        
                        if metrics is not None:
                            for name, func in metrics.items():
                                metrics_val[f"validation_{name}"] += func(y_pred, y_true)

# #################################################
#                         break
# #################################################

                    validation_loss /= validation_data_len
                    if metrics is not None:
                        for name in metrics:
                            metrics_val[f"validation_{name}"] /= validation_data_len

                # Learning rate scheduling
                if lr_scheduler is not None: 
                    lr_scheduler.step(validation_loss)

            # Logging
                if verbose:
                    print(f"epoch {epoch:4d} | train loss {train_loss:.4f} | validation loss {validation_loss:.4f}", end="")
                    if metrics is not None: 
                        for name in metrics:
                            print(f" | train {name} {metrics_val[f'train_{name}']:.4f} | validation {name} {metrics_val[f'validation_{name}']:.4f}", end="")
                    print()

                # Store validation metrics
                self.fit_history["validation_loss"] = self.fit_history.get("validation_loss", []) + [validation_loss]
                history["validation_loss"] = history.get("validation_loss", []) + [validation_loss]
                if metrics is not None: 
                    for name in metrics:
                        self.fit_history[f"validation_{name}"] = self.fit_history.get(f"validation_{name}", []) + [metrics_val[f"validation_{name}"]]
                        history[f"validation_{name}"] = history.get(f"validation_{name}", []) + [metrics_val[f"validation_{name}"]]

            # if validation_dataloader is None:
            else:
                if verbose:
                    print(f"epoch {epoch:4d} | train loss {train_loss:.4f}", end="")
                    if metrics is not None:
                        for name in metrics:
                            print(f" | train {name} {metrics_val[f'train_{name}']:.4f}", end="")
                    print()
            
            # Store training metrics
            self.fit_history["train_loss"] = self.fit_history.get("train_loss", []) + [train_loss]
            history["train_loss"] = history.get("train_loss", []) + [train_loss]
            if metrics is not None: 
                for name in metrics:
                    self.fit_history[f"train_{name}"] = self.fit_history.get(f"train_{name}", []) + [metrics_val[f"train_{name}"]]
                    history[f"train_{name}"] = history.get(f"train_{name}", []) + [metrics_val[f"train_{name}"]]

            for callback in callbacks:
                # if isinstance(callback, EarlyStopper):
                if callback(self, history, epoch):
                    return history

        return history



    def get_size_in_bits(self):
        size = 0
        for layer in self.layers.values():
            size += layer.get_size_in_bits()
        return size


    @torch.inference_mode()
    def evaluate(
        self, 
        data_loader: data.DataLoader, 
        metric_fun: Callable, 
        device: str = "cpu"
    ) -> float:
        """Evaluate model accuracy on given dataset
        
        Args:
            data_loader: Data loader for evaluation
            device: Device to run on
            
        Returns:
            Accuracy percentage
        """
        self.eval()
        metric_val = 0
        data_len = 0

        for X, y_true in tqdm(data_loader):
            X = X.to(device)
            y_true = y_true.to(device)
            y_pred = self(X)
            metric_val += metric_fun(y_pred, y_true)
            data_len += X.size(0)

# #################################################
            # break
# #################################################

        return metric_val / data_len
    
    
    def prepare_compression(
            self,
            config:OrderedDict[str, Dict[str, Union[int,float, Dict[str, float]]]],
            input_shape:torch.Size,
    ):
        
        model = copy.deepcopy(self)

        def is_config_valid():
            for configuration_type in config.keys():
                if configuration_type == "prune_channel":
                    prune_channel_config = config.get("prune_channel")

                    sparsity = prune_channel_config.get("sparsity")

                    if isinstance(sparsity, float):
                        layer_sparsity = sparsity
                        sparsity = OrderedDict()
                        for name in model.layers.keys():
                            sparsity[name] = layer_sparsity

                    elif isinstance(sparsity, OrderedDict):
                        for name in model.layers.keys():
                            if name not in sparsity:
                                sparsity[name] = 0.

                    else:
                        raise TypeError(f"prune sparsity has to be of type of float or orderdict not {type(sparsity)}!")
                    
                    prune_channel_config["sparsity"] = sparsity

                elif configuration_type == "quantization":
                    quantization_config = config.get("quantization")

                    if quantization_config["type"] != DYNAMIC_QUANTIZATION_PER_TENSOR:
                        raise ValueError(f"Invalid quantization type")
                    
                    if quantization_config["bitwidth"] is not None and quantization_config["bitwidth"] > 8:
                        raise ValueError(f"Invalid quantization bitwidth")
                    
                else:
                    raise ValueError(f"Invalid configuration type of {configuration_type}")
                
            print("Config", config)
                    
            return True
        
        if not is_config_valid():
            raise ValueError("Invalid compression configuration!")
        
        model.__dict__["_dmc"]["compression_config"] = config 
        model.__dict__["_dmc"]["input_shape"] = input_shape  
                
        for compression_type, compression_type_param in config.items():
            if compression_type == "prune_channel":
                model.prepare_prune_channel(
                    sparsity=compression_type_param["sparsity"]
                )
            elif compression_type == "quantization":
                model.prepare_quantization(
                    bitwidth = compression_type_param["bitwidth"],
                    type = compression_type_param["type"]
                )
            else:
                raise NotImplementedError(f"Compression of type {compression_type} not implemented!")

        return model
    

    def apply_compression(self):
        if "compression_config" not in self.__dict__["_dmc"]:
            raise AttributeError("Cannot apply compression before prepering for compression.")
        
        for compression_type in self.__dict__["_dmc"]["compression_config"].keys():
            if compression_type == "prune_channel":
                self.apply_prune_channel()
            elif compression_type == "quantization":
                self.apply_quantization()
            else:
                raise NotImplementedError(f"Compression of type {compression_type} not implemented!")
            
            # print("compression", compression_type)

        return
    
    def compress(
            self,
            config,
            input_shape
    ):
        model = self.prepare_compression(config, input_shape)
        model.apply_compression()

        return model
    
    

    def prepare_prune_channel(
            self, 
            sparsity: Dict[str, float], 
            metric: str = "l2"
        ) -> None:
        """Create pruned version of model
        
        Args:
            sparsity: Either single value or dict of sparsities per layer
            metric: Pruning metric ('l1', 'l2', etc.)
            
        Returns:
            New model instance with pruned channels
        """

        keep_prev_channel_index = None
            
        input_shape = self.__dict__["_dmc"]["input_shape"][1:]

        # Prune all layers except last
        for name, layer in list(self.layers.items())[:-1]:

            keep_prev_channel_index = layer.prepare_prune_channel(
                sparsity[name], keep_prev_channel_index, input_shape,
                is_output_layer=False, metric=metric
            )
            _, input_shape = layer.get_output_tensor_shape(input_shape)

        # Prune last layer
        name, layer = list(self.layers.items())[-1]
        keep_prev_channel_index = layer.prepare_prune_channel(
            sparsity[name], keep_prev_channel_index, input_shape,
            is_output_layer=True, metric=metric
        )
        return 
    
    def apply_prune_channel(
            self
    ):
        for layer in self.layers.values():
            layer.apply_prune_channel()
        return




    def prepare_quantization(
        self, 
        bitwidth,
        type,
        input_batch_real:Optional[torch.Tensor] = None
    ):
        
        # if bitwidth in None:
        #     return

        self.__dict__["_dmc"]["quantization"] = dict()

        
        if type == DYNAMIC_QUANTIZATION_PER_TENSOR:  
            for layer in self.layers.values():
                layer.prepare_quantization(bitwidth, type)

        elif type == STATIC_QUANTIZATION_PER_TENSOR: 
            if input_batch_real is None:
                raise ValueError(f"pass a calibration prarmeter")    
            input_scale, input_zero_point = get_quantize_scale_zero_point_per_tensor_assy(input_batch_real, bitwidth)
            input_batch_quant = quantize_per_tensor_assy(input_batch_real, input_scale, input_zero_point, bitwidth)

            self.__dict__["_dmc"]["quantization"]["input_scale"] = input_scale
            self.__dict__["_dmc"]["quantization"]["input_zero_point"] = input_zero_point
       
            for layer in self.layers.values():
                input_batch_real, input_batch_quant, \
                input_scale, input_zero_point = layer.prepare_quantization(
                    input_batch_real, input_batch_quant,
                    input_scale, input_zero_point,
                    bitwidth,
                )
            
            self.__dict__["_dmc"]["quantization"]["output_scale"] = input_scale
            self.__dict__["_dmc"]["quantization"]["output_zero_point"] = input_zero_point


    def apply_quantization(
        self, 
    ):
        if "quantization" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
    
        for layer in self.layers.values():
            layer.apply_dynamic_quantization_per_tensor()

        # print("I fuclkalkf")
        # if type == DYNAMIC_QUANTIZATION_PER_TENSOR:            
        #     print("<<<<<<<<<<<I fuclkalkf")
        #     for layer in self.layers.values():
        #         layer.apply_dynamic_quantization_per_tensor()
        # elif type == STATIC_QUANTIZATION_PER_TENSOR:            
        #     for layer in self.layers.values():
        #         layer.apply_static_quantization_per_tensor()




    def get_max_workspace_arena(self) -> tuple:
        """Calculate memory requirements for C deployment by running sample input
        
        Returns:
            Tuple of (max_even_size, max_odd_size) workspace requirements
        """
        # Create random input tensor based on model's expected input shape

        input_shape = self.__dict__["_dmc"]["input_shape"][1:]
        
        # max_output_even_size = input_shape.numel()
        max_output_even_size = 0
        max_output_odd_size = 0
        
        output_shape = input_shape
        
        # Track maximum tensor sizes at even/odd layers
        for i, layer in enumerate(self.layers.values()):
            max_layer_shape, output_shape = layer.get_output_tensor_shape(input_shape)
            if (i % 2 == 0):
                max_output_even_size = max(max_output_even_size, input_shape.numel(), max_layer_shape.numel())
            else:
                max_output_odd_size = max(max_output_odd_size, input_shape.numel(), max_layer_shape.numel())
        
            input_shape = output_shape
            # print(max_layer_shape, output_shape, i, layer.__class__.__name__)
        if len(self.layers) % 2 == 0:
            max_output_even_size = max(max_output_even_size, output_shape.numel())
        else:
            max_output_odd_size = max(max_output_odd_size, output_shape.numel())

        return max_output_even_size, max_output_odd_size


    @torch.no_grad()
    def convert_to_c(self, var_name: str, src_dir: str = "./", include_dir:str = "./") -> None:
        """Generate C code for deployment
        
        Args:
            var_name: Base name for generated files
            dir: Output directory for generated files
        """
        def write_str_to_c_file(file_str: str, file_name: str, dir: str):
            """Helper to write string to file"""
            with open(path.join(dir, file_name), "w") as file:
                file.write(file_str)
        
        # Initialize file contents
        header_file = f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n"
        header_file += "#include <stdint.h>\n#include \"deep_microcompression.h\"\n\n\n"

        definition_file = f"#include \"{var_name}.h\"\n\n"
        param_definition_file = f"#include \"{var_name}.h\"\n\n"
    
        # Calculate workspace requirements
        max_output_even_size, max_output_odd_size = self.get_max_workspace_arena()

        # Configure workspace based on quantization
        if getattr(self, "quantization_type", QUANTIZATION_NONE) != STATIC_QUANTIZATION_PER_TENSOR:
            workspace_header = (
                f"#define MAX_OUTPUT_EVEN_SIZE {max_output_even_size}\n"
                f"#define MAX_OUTPUT_ODD_SIZE {max_output_odd_size}\n"
                f"extern float workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )
            workspace_def = f"float workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
        else:
            workspace_header = (
                f"#define MAX_OUTPUT_EVEN_SIZE {max_output_even_size}\n"
                f"#define MAX_OUTPUT_ODD_SIZE {max_output_odd_size}\n"
                f"extern int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"
            )
            workspace_def = f"int8_t workspace[MAX_OUTPUT_EVEN_SIZE + MAX_OUTPUT_ODD_SIZE];\n\n"

        header_file += workspace_header
        definition_file += workspace_def

        # Generate layer declarations
        layers_header = (
            f"#define LAYERS_LEN {len(self.layers)}\n"
            f"extern Layer* layers[LAYERS_LEN];\n\n"
            f"extern Sequential {var_name};\n\n"
        )
        layers_def = (
            f"{self.__class__.__name__} {var_name}(layers, LAYERS_LEN, workspace, MAX_OUTPUT_EVEN_SIZE);\n"
            f"\nLayer* layers[LAYERS_LEN] = {{\n"
        )
        
        # Convert each layer to C
        input_shape = self.input_shape[1:]
        for layer_name, layer in self.layers.items():
            if hasattr(layer, "convert_to_c"):
                layers_def += f"    &{layer_name},\n"

                layer_header, layer_def, layer_param_def = layer.convert_to_c(layer_name, input_shape)
                layers_header += layer_header

                param_definition_file += layer_param_def
                definition_file += layer_def 

                _, input_shape = layer.get_output_tensor_shape(input_shape)
            else:
                raise RuntimeError(f"The cpp conversion of {layer.__class__.__name__} has not been implemented!")
        
        layers_def += "};\n"
        definition_file += layers_def
        header_file += layers_header
        header_file += f"\n#endif //{var_name.upper()}_h\n"

        # Write files
        write_str_to_c_file(header_file, f"{var_name}.h", include_dir)
        write_str_to_c_file(definition_file, f"{var_name}_def.cpp", src_dir)
        write_str_to_c_file(param_definition_file, f"{var_name}_params.cpp", src_dir)


#################################################################################################################
        # Generate test input data
        # _, test_input_def = convert_tensor_to_bytes_var(self.test_input, "test_input", getattr(self, "quantization_bitwidth", 8))
        

        if hasattr(self, "quantization_type") and (getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR
                                                   or getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL):
            test_input_def = f"\nconst uint8_t test_input[] = {{\n"
            for line in torch.split(quantize_per_tensor_assy(
                    self.test_input, self.input_scale, self.input_zero_point, self.quantization_bitwidth).flatten(), 8):
                test_input_def += "    " + ", ".join(
                    [f"0x{b:02X}" for val in line for b in int8_to_bytes(val)]
                ) + ",\n"
            test_input_def += "};\n"
        else:
            test_input_def = f"\nconst float test_input[] = {{\n"
            for line in torch.split(self.test_input.flatten(), 8):
                test_input_def += "    " + ", ".join(
                    [f"{val:.4f}" for val in line]
                ) + ",\n"
            test_input_def += "};\n"

    
        with open(path.join(include_dir, f"{var_name}_test_input.h"), "w") as file:
            file.write(test_input_def)


#################################################################################################################

        return
    
    @torch.no_grad
    def test(self, device:str = "cpu"):
        # self.cpu()
        # if hasattr(self, "quantization_type") and (getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR
        #                                            or getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL):
        #     return self(quantize_per_tensor_assy(
        #             self.test_input.clone(), self.input_scale, self.input_zero_point, self.quantization_bitwidth))
        
        return self(self.test_input.clone().to(device))



















































    # def forward(self, input):
    #     """Forward pass with quantization support
        
    #     Args:
    #         input: Input tensor
            
    #     Returns:
    #         Output tensor after passing through all layers
    #     """
    #     # Store input shape and sample test input (for later code generation)
    #     setattr(self, "input_shape", input.size())


    #     ################################################################################################
    #     # Saving the a test input data
    #     import random
    #     index = random.randint(0, input.size(0)-1)
    #     # index = 0
    #     # print(f"Using this {index}")

    #     test_input = input[index].unsqueeze(dim=0).cpu()
    #     # if not hasattr(self, "test_input"):
    #     setattr(self, "test_input", test_input)
        
    #     # if hasattr(self, "quantization_type") and (getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR
    #     #                                            or getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL):
    #     #     if not hasattr(self, "test_input_quant"):
    #     #         setattr(self, "test_input_quant", quantize_per_tensor_assy(
    #     #             test_input, self.input_scale, self.input_zero_point, self.quantization_bitwidth)
    #     #         )

    #     ################################################################################################
        
    #     # Apply quantization if configured
    #     # if hasattr(self, "quantization_type") and getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #     input = quantize_per_tensor_assy(input, self.input_scale, self.input_zero_point, self.quantization_bitwidth)
    #     # elif hasattr(self, "quantization_type") and getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
    #     #     input = quantize_per_tensor_assy(input, self.input_scale, self.input_zero_point, self.quantization_bitwidth)

    #     # Pass through all layers
    #     for i, layer in enumerate(self):
    #         # print(f"Layer {i} ({layer.__class__.__name__}): input shape = {input.shape} kernel_size {getattr(layer, "kernel_size", "nan")} stride {getattr(layer, "stride", "nan")} padding {getattr(layer, "padding", "nan")}")
    #         input = layer(input)
                
    #     return input



    # @torch.no_grad()
    # def revert_param(self):
    #     """Reset all parameters and quantization states"""
    #     for layer in self.layers.values():
    #         if hasattr(layer, "reset_buffer"): 
    #             layer.reset_buffer()
        
    #     if hasattr(self, "quantization_type"): 
    #         setattr(self, "quantization_type", QUANTIZATION_NONE)
    #     return

    

    def get_weight_distributions(self, bins=256) -> Dict[str, Optional[torch.Tensor]]:
        """Get weight histograms for all layers
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Dictionary mapping layer names to weight histograms
        """
        weight_dist = dict()
        for name, layer in self.layers.items():
            if hasattr(layer, "weight"): 
                weight_dist[name] = torch.histogram(layer.weight.cpu(), bins=bins)
            else: 
                weight_dist[name] = None
        return weight_dist
    



    # def set_compression_parameters(self) -> None:
    #     for layer in self.layers.values():
    #         if hasattr(layer, "set_compression_parameters"):
    #             layer.set_compression_parameters()

    #     return


    @torch.no_grad()
    def get_layers_prune_channel_sensity(self, data_loader: data.DataLoader, 
                                       sparsities: Dict[str, List], 
                                       metric: str = "l2", 
                                       device="cpu") -> Dict[str, List[float]]:
        """Analyze pruning sensitivity for each layer
        
        Args:
            data_loader: Data for evaluation
            sparsities: Dictionary of sparsity values to test per layer
            metric: Pruning metric ('l1', 'l2', etc.)
            device: Device to run on
            
        Returns:
            Dictionary of accuracy results for each sparsity level
        """
        history = dict()
        default_config = dict()
        for name in self.layers.keys():
            default_config[name] = 0.2

        # Test each layer's sensitivity to pruning
        for name in self.layers.keys():
            history[name] = []
            for sparsity in tqdm(sparsities[name], desc=f"Pruning {name}"):
                config = default_config.copy()
                config[name] = sparsity
                model = self.prune_channel(config, metric=metric)
                history[name] += [{sparsity : model.evaluate(data_loader, device=device)}]

        # Plot results
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
    
        

    @torch.no_grad()
    def get_dynamic_quantize_per_tensor_sensity(self, data_loader: data.DataLoader, 
                                              bitwidths: Iterable[int], 
                                              device: str = "cpu") -> Dict[float, List[float]]:
        """Analyze sensitivity to different quantization bitwidths (per-tensor)
        
        Args:
            data_loader: Data for evaluation
            bitwidths: List of bitwidths to test
            device: Device to run on
            
        Returns:
            Dictionary of accuracy results for each bitwidth
        """
        history = dict()

        for bitwidth in bitwidths:
            model = self.dynamic_quantize_per_tensor(bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)


        return history

    @torch.no_grad()
    def dynamic_quantize_per_tensor(self, bitwidth: int = 8):
        """Apply dynamic per-tensor quantization to model
        
        Args:
            bitwidth: Number of bits to use for quantization (max 8)
            
        Returns:
            New quantized model instance
        """
        assert bitwidth <= 8

        # model = copy.deepcopy(self)
        setattr(self, "quantization_bitwidth", bitwidth)
        setattr(self, "quantization_type", DYNAMIC_QUANTIZATION_PER_TENSOR)

        for layer in self.layers.values():
            # if hasattr(layer, "dynamic_quantize_per_tensor"):
                layer.dynamic_quantize_per_tensor(bitwidth)

        return 

    @torch.no_grad()
    def get_dynamic_quantize_per_channel_sensity(self, data_loader: data.DataLoader, 
                                               bitwidths: Iterable[float], 
                                               device: str = "cpu") -> Dict[float, List[float]]:
        """Analyze sensitivity to different quantization bitwidths (per-channel)
        
        Args:
            data_loader: Data for evaluation
            bitwidths: List of bitwidths to test
            device: Device to run on
            
        Returns:
            Dictionary of accuracy results for each bitwidth
        """
        history = dict()

        for bitwidth in bitwidths:
            model = self.dynamic_quantize_per_channel(bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history

    @torch.no_grad()
    def dynamic_quantize_per_channel(self, bitwidth: int = 8):
        """Apply dynamic per-channel quantization to model
        
        Args:
            bitwidth: Number of bits to use for quantization (max 8)
            
        Returns:
            New quantized model instance
        """
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", DYNAMIC_QUANTIZATION_PER_CHANNEL)
        setattr(model, "quantization_bitwidth", bitwidth)

        for layer in model.layers.values():
            if hasattr(layer, "dynamic_quantize_per_channel"):
                layer.dynamic_quantize_per_channel(bitwidth)

        return model

    @torch.no_grad()
    def get_static_quantize_per_tensor_sensity(self, input_batch_real: torch.Tensor, 
                                             data_loader: data.DataLoader, 
                                             bitwidths: Iterable[float], 
                                             device: str = "cpu") -> Dict[float, List[float]]:
        """Analyze sensitivity to different static quantization bitwidths (per-tensor)
        
        Args:
            input_batch_real: Example input data for calibration
            data_loader: Data for evaluation
            bitwidths: List of bitwidths to test
            device: Device to run on
            
        Returns:
            Dictionary of accuracy results for each bitwidth
        """
        history = dict()

        for bitwidth in bitwidths:
            model = self.static_quantize_per_tensor(input_batch_real, bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history

    @torch.no_grad()
    def static_quantize_per_tensor(self, input_batch_real: torch.Tensor, bitwidth: int = 8):
        """Apply static per-tensor quantization to model
        
        Args:
            input_batch_real: Example input data for calibration
            bitwidth: Number of bits to use for quantization (max 8)
            
        Returns:
            New quantized model instance
        """
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)
        setattr(model, "quantization_bitwidth", bitwidth)

        # Calculate quantization parameters
        input_scale, input_zero_point = get_quantize_scale_zero_point_per_tensor_assy(input_batch_real, bitwidth)
        input_batch_quant = quantize_per_tensor_assy(input_batch_real, input_scale, input_zero_point, bitwidth)

        model.register_buffer("input_scale", input_scale)
        model.register_buffer("input_zero_point", input_zero_point)

        # Quantize each layer
        for layer in model.layers.values():
            if hasattr(layer, "static_quantize_per_tensor"):
                input_batch_real, input_batch_quant, \
                input_scale, input_zero_point = layer.static_quantize_per_tensor(
                    input_batch_real, input_batch_quant,
                    input_scale, input_zero_point,
                    bitwidth,
                )
            else:
                raise AttributeError(f"Static Quantization Per Tensor not implemented for {layer.__class__.__name__}!!!")

        model.register_buffer("output_scale", input_scale)
        model.register_buffer("output_zero_point", input_zero_point)

        return model


    @torch.no_grad()
    def get_static_quantize_per_channel_sensity(self, input_batch_real: torch.Tensor, 
                                              data_loader: data.DataLoader, 
                                              bitwidths: Iterable[float], 
                                              device: str = "cpu") -> Dict[float, List[float]]:
        """Analyze sensitivity to different static quantization bitwidths (per-channel)
        
        Args:
            input_batch_real: Example input data for calibration
            data_loader: Data for evaluation
            bitwidths: List of bitwidths to test
            device: Device to run on
            
        Returns:
            Dictionary of accuracy results for each bitwidth
        """
        history = dict()

        for bitwidth in bitwidths:
            model = self.static_quantize_per_channel(input_batch_real, bitwidth)
            history[bitwidth] = model.evaluate(data_loader, device=device)
            print(history)

        return history

    @torch.no_grad()
    def static_quantize_per_channel(self, input_batch_real: torch.Tensor, bitwidth: int = 8):
        """Apply static per-channel quantization to model
        
        Args:
            input_batch_real: Example input data for calibration
            bitwidth: Number of bits to use for quantization (max 8)
            
        Returns:
            New quantized model instance
        """
        assert bitwidth <= 8

        model = copy.deepcopy(self)
        setattr(model, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)
        setattr(model, "quantization_bitwidth", bitwidth)

        # Calculate quantization parameters
        input_scale, input_zero_point = get_quantize_scale_zero_point_per_tensor_assy(input_batch_real, bitwidth)
        input_batch_quant = quantize_per_tensor_assy(input_batch_real, input_scale, input_zero_point, bitwidth)

        model.register_buffer("input_scale", input_scale)
        model.register_buffer("input_zero_point", input_zero_point)

        # Quantize each layer
        for layer in model.layers.values():
            if hasattr(layer, "static_quantize_per_channel"):
                input_batch_real, input_batch_quant, \
                input_scale, input_zero_point = layer.static_quantize_per_channel(
                    input_batch_real, input_batch_quant,
                    input_scale, input_zero_point,
                    bitwidth,
                )
            else:
                raise AttributeError(f"Static Quantization Per Channel not implemented for {layer.__class__.__name__}!!!")

        model.register_buffer("output_scale", input_scale)
        model.register_buffer("output_zero_point", input_zero_point)

        return model
#