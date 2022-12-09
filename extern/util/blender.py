import numpy as np
import torch

class Blender:
    def __init__(self, base, styles, device):
        self.base = base
        self.styles = styles
        self.device = device
        
    def blend_models(self, mix):
        resolutions =  [4*2**x for x in range(int(np.log2(1024)-1))]

        #start with lower model and add weights above
        dest_dict = self.base.state_dict()
        src_dicts = [style.state_dict() for style in self.styles]
        # params_src = styles.named_parameters()

        for name, _ in self.base.named_parameters():
            res_index = None
            for i, res in enumerate(resolutions):
                if f'synthesis.b{res}' in name:
                    res_index = i
                    break
            if res_index and not ('mapping' in name):
                data=src_dicts[0][name].data*mix[0][res_index]
                for i in range(1, len(src_dicts)):
                    data+=src_dicts[i][name].data*mix[i][res_index]
                dest_dict[name].data.copy_(data)

        base_dict = self.base.state_dict()
        base_dict.update(dest_dict) 
        self.base.load_state_dict(dest_dict)

    def generate(self, latent_vector):
        img = self.base.synthesis(torch.from_numpy(latent_vector).to(self.device), noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_numpy = img[0].cpu().numpy()
        return img_numpy