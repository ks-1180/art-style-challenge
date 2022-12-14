import torch
import dnnlib
import legacy
import os
import pandas as pd
import numpy as np
import cv2
import copy
import mixes
class Generator:
    def __init__(self, base_model, styles_dir, resoltuion):
        self.resolution = resoltuion
        self.truncation_psi = 0.5

        self.stlyes_paths = [f'{styles_dir}/{p}' for p in os.listdir(f'{styles_dir}/')]
        self.stlyes_paths.sort()
        self.stlyes_names = []
        for source in self.stlyes_paths:
            name = os.path.splitext(os.path.basename(source))
            self.stlyes_names.append(name)

        #self.evals = []
        #for source in self.stlyes_paths:
        #    name = os.path.splitext(os.path.basename(source))
        #    e = evaluator.Evaluator(name, base_model, )

        self.G_ffhq = self.generate_network(base_model)
        self.G_styles = [self.generate_network(network) for network in self.stlyes_paths]
        
    def generate_network(self, path):
        device = torch.device('cuda')
        f = dnnlib.util.open_url(path)
        return legacy.load_network_pkl(f)['G_ema'].to(device)

    def generate_image(self, z, G, psi=float):
        w = G.mapping(z, None, truncation_psi = psi, truncation_cutoff=8)
        img = G.synthesis(w, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

    def generate_seed(self):
        z = torch.randn([1, self.G_ffhq.z_dim]).cuda()
        img = self.generate_image(z, self.G_ffhq, self.truncation_psi)
        return img, z

    def generate_art_styles(self, seed=None):
        if seed == None:
            seed = self.generate_seed()

        result = self.generate_image(seed, self.G_ffhq, self.truncation_psi)
        for G_new, name in list(zip(self.G_styles, self.stlyes_names)):
            G_blend = copy.deepcopy(self.G_ffhq)
            print(name[0])
            mix = mixes.Mixes[name[0]].value
            styles = [self.G_ffhq, G_new]
            self.__mix(G_blend, styles, mix)
            img = self.generate_image(seed, G_blend, psi=self.truncation_psi)
            result = np.hstack((result, img))
        return result
    

    def __mix(self, base, styles, mix):
        resolutions =  [4*2**x for x in range(int(np.log2(self.resolution)-1))]

        dest_dict = base.state_dict()
        src_dicts = [style.state_dict() for style in styles]

        for name, _ in base.named_parameters():
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

        base_dict = base.state_dict()
        base_dict.update(dest_dict) 
        base.load_state_dict(dest_dict)
