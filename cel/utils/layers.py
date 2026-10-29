import torch
from typing import Any, Dict, Optional, Union
import numpy as np
from torch.autograd import Variable
from .gradients import get_gradients

def getLayers(module, name="model", layer_names=[]):
    layers={}
    for i in module.named_children():
        if(dict(getattr(module,i[0]).named_children())=={}):
            layers[name+"."+i[0]]=getattr(module,i[0])
            continue
        else:
            layers.update(getLayers(getattr(module,i[0]), name=name+"."+i[0]))
    
    if layer_names != []:
        layers = dict((k,layers[k]) for k in layer_names) #TODO: summarize in ECLAD run code
    return layers

def get_nth_differentiable_layer(model, test_input, n=1):
    layers = getLayers(model)
    differentiable_layers = []
    other_layers=[]
    current_input = test_input.clone().detach().requires_grad_(True)
    for name, module in layers.items():
        output = module(current_input)
        is_new_op = (output.grad_fn is not None) and (output is not current_input)
        if is_new_op and output.requires_grad:
            differentiable_layers.append({name: module})
        else:
            other_layers.append({name: module})
        if len(differentiable_layers) == n:
            return differentiable_layers[-1], other_layers
        current_input = output
    raise RuntimeError(f"Less than {n} differentiable layers found!")

def nested_hook(layer_id:str, activation_store:dict):
    def myhook(module, input, output):
        activation_store[layer_id]=output
    return myhook

def getActivations(model, layers, input):
    activation_store={}
    #Set hooks
    handles=[]
    for key, value in layers.items():
        handles.append(value.register_forward_hook(nested_hook(key, activation_store)))

    #Forward pass
    outputs = model(input)

    #Remove hooks
    for handle in handles:
        handle.remove()
    return activation_store, outputs


def reshape_activations(activations:dict, input_shape, input_dimension: Optional[str]=None, device='cpu'):
    reshaped=torch.empty(0).to(device)
    activations = {k: v.to(device) for k, v in activations.items()}
    # determine dimension if not given
    if input_dimension == None:
        if(type(input_shape)==int): input_dimension='1D'
        elif(len(input_shape)==2): input_dimension='2D'
        else: print('input dimension not valid')
    # choose proper interpolaation depending on dimensionality
    if(input_dimension=='1D' or input_dimension=='timeseries'): 
        m=torch.nn.Upsample(input_shape, mode='linear') #TODO: nearest?
    elif(input_dimension=='2D' or input_dimension=='images'): 
        m=torch.nn.Upsample(input_shape, mode='bilinear')
    # reshape activations
    for _, value in activations.items():
        reshaped=torch.cat((reshaped, m(value)), dim=1)
    return reshaped

def getMaxActivationShape(test_activations, dimension="timeseries"):
        """Get the maximum activation shape from the test layers.
        Args:
            test_activations (dict): Dictionary of test activations.
        Returns:    
            int or list: Maximum activation shape.
        """
        if dimension == ("1D") or dimension == ("timeseries"):
            # Activations of shape batch_size x channels x length
            act_shape = 0
            for act in test_activations.values():
                if act.shape[-1] > act_shape:
                    act_shape = act.shape[-1]
        elif dimension == ("2D") or dimension == ("images"):
            s1, s2 = 0, 0
            for act in test_activations.values():
                s1 = max(s1, act.shape[-2])
                s2 = max(s2, act.shape[-1])
            act_shape = [s1, s2]
        return act_shape


def compute_RF_numerical(model,layer_name, in_shape):
    '''
    model: Pytorch network
    layer_name: name of the layer for which to compute the EFR (list with single name)
    in_shape: shape of the input of the NN

    Adapted from: https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb
    '''

    model = model.to('cpu')
    model.eval()
    img_ = Variable(torch.ones(in_shape).float(),requires_grad=True).to('cpu')
    test_layers = getLayers(model, layer_names=layer_name)
    out_cnn, _ = getActivations(model, test_layers, img_)
    for n in range(out_cnn[layer_name[0]].shape[-1]//2):
        neuron = out_cnn[layer_name[0]][:,:, out_cnn[layer_name[0]].shape[-1]//2 + n] #only for CShap so far
        grad_np = get_gradients(neuron, img_)[0,0]
        idx_nonzeros=np.where(grad_np!=0)
        if idx_nonzeros[0].size>0:
            return [int(np.max(idx)-np.min(idx)+1) for idx in idx_nonzeros]

    return None


def layer_correspondence(orig_len, target_len):
    divisor = target_len / orig_len
    list1 = np.zeros(orig_len)
    for k in range(target_len):
        list1[int(k / divisor)] = k
    return list1


def zero_upsample(input_tensor, target_size):
    """
    Upsamples an input tensor to a target size (c, d) using zero insertion instead of interpolation.

    Args:
        input_tensor (torch.Tensor): Input tensor containing labels of shape (B, H, W) or (B, W). No channels expected.
        target_size (tuple): Target output size (target_H, target_W). 

    Returns:
        torch.Tensor: Upsampled tensor with zeros inserted.
    """
    if input_tensor.dim() == 2:  # If input is (B, W), add batch and channel dims
        B, W = input_tensor.shape

        target_W = target_size

        corr_W = layer_correspondence(W, target_W)

        # Create an upsampled tensor filled with zeros
        upsampled = torch.zeros(
            (B, target_W), dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Fill the upsampled tensor with original values at appropriate locations
        upsampled[:, corr_W] = input_tensor
    elif input_tensor.dim() == 3:
        B, H, W = input_tensor.shape

        target_H, target_W = target_size

        corr_H = layer_correspondence(H, target_H)
        corr_W = layer_correspondence(W, target_W)

        # Create an upsampled tensor filled with zeros
        upsampled = torch.zeros(
            (B, target_H, target_W),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        # Fill the upsampled tensor with original values at appropriate locations
        upsampled[:, corr_H, corr_W] = input_tensor

    return upsampled.squeeze()
