import gc
import torch

for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            if (obj.nelement()*obj.element_size() > 1000000):
                print(obj.type(), obj.size(), obj.nelement()*obj.element_size() / 1000000.0)
    except:
        pass