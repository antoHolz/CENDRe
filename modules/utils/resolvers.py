import torch

def determine_balanced(task: str):
    if task == "classification":
        return True
    elif task == "regression":
        return False
    assert False