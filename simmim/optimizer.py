from torch import optim as optim
from functools import partial

def build_pretrain_optimizer(epsilon, betas, base_lr, weight_decay, model):

    parameters = get_pretrain_param_groups(model)

    optimizer = None
    optimizer = optim.AdamW(parameters, eps = epsilon, betas = betas,
                            lr = base_lr, weight_decay = weight_decay)

    return optimizer

def get_pretrain_param_groups(model):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def build_finetune_optimizer(layer_decay, base_lr, epsilon, betas, depth, weight_decay, model):

    get_layer_func = partial(get_vit_layer, num_layers = depth + 2)
    
    scales = list(layer_decay ** i for i in reversed(range(depth + 2)))

    parameters = get_finetune_param_groups(
        model, base_lr, weight_decay,
        get_layer_func, scales)
    
    optimizer = None

    optimizer = optim.AdamW(parameters, eps= epsilon, betas = betas,
                            lr = base_lr, weight_decay = weight_decay)

    return optimizer


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1

def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin