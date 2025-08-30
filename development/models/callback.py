from typing import Union, Callable, Dict, List

from .sequential import Sequential

class EarlyStopper:

    def __init__(
        self, 
        metric_name: str,
        min_valid_diff:float,
        patience:int,
        mode:str,
        restore_best_state_dict:bool = False
    ):
        self.metric_name = metric_name
        self.min_valid_diff = min_valid_diff
        self.patience = patience

        assert mode == "min" or mode == "max", "Mode must be either \"min\" or \"max\""
        self.mode = mode
        self.restore_best_state_dict = restore_best_state_dict

        self.best_metric_val = None
        self.best_epoch = 0
        self.best_state_dict = None


    def __call__(
            self,
            model:Sequential,
            history:Dict[str, List[float]],
            epoch:int
    ):
        assert self.metric_name in history.keys(), f"Metric {self.metric_name} is not the model training dictionary, {history.keys()}!"

        if self.best_metric_val is None or \
           (self.mode == "min" and (self.best_metric_val - history[self.metric_name][-1]) > self.min_valid_diff) or\
           (self.mode == "max" and (history[self.metric_name][-1] - self.best_metric_val) > self.min_valid_diff):
                self.best_metric_val = history[self.metric_name][-1]
                self.best_epoch = epoch

                if self.restore_best_state_dict: 
                    self.best_state_dict = model.state_dict()
                return False
        
        if epoch - self.best_epoch >= self.patience:
            print(f"Stopping Training of {model.__class__.__name__} with at {self.best_epoch} epoch with best {self.metric_name} = {self.best_metric_val}")
            
            if self.restore_best_state_dict and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict, strict=True)
            return True
            
        return False
            