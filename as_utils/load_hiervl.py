import torch
# import model.hiervl_model as module_arch
import model.hiervl_model as module_arch
import json

# adapted from hiervl

def load_hiervl(model_path):
    """
    Load from saved checkpoints

    :param model_path: Checkpoint path to be loaded
    """
    
    model_path = str(model_path)
    print("Loading checkpoint: {} ...".format(model_path))

    checkpoint = torch.load(model_path, map_location='cpu')

    config = torch.load("/nfs/wattrel/data/md0/datasets/state_aware/results/EgoClip_CF/models/0226_23_46_03/ckpt_18b_1e5_epoch7_correct.pth", map_location='cpu')['config']
    config['arch']['type'] = "FrozenInTime"
    model = config.initialize('arch', module_arch)

    state_dict = checkpoint['state_dict']

    load_state_dict_keys = list(state_dict.keys())
    curr_state_dict_keys = list(model.state_dict().keys())
    redo_dp = False
    if not curr_state_dict_keys[0].startswith('module.') and load_state_dict_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_state_dict_keys[0].startswith('module.') and not load_state_dict_keys[0].startswith('module.'):
        redo_dp = True
        undo_dp = False
    else:
        undo_dp = False

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif False: # elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False

    # load optimizer state from checkpoint only when optimizer type is not changed.

    print("Checkpoint loaded.")

    return model

if __name__ == "__main__":
    # model = model_load("/nfs/wattrel/data/md0/datasets/state_aware/hievl_sa.pth")
    # x = {"video": torch.randn(1, 16, 3, 224, 224)}
    # y = model(x)
    # print(y.shape)
    raise Exception("I should not be main")