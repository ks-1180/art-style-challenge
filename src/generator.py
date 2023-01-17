import torch
import dnnlib
import legacy
import os
import pandas as pd
import numpy as np
import cv2
import copy
import mixes
import projector as projector
import PIL.Image

class Generator:
    def __init__(self, base_model, styles_dir, resoltuion):
        self.resolution = resoltuion
        self.truncation_psi = 0.5
        self.device = torch.device('cuda')
        self.styles_dir = styles_dir
        self.G_ffhq = self.generate_network(base_model)
        
    def generate_network(self, path):
        f = dnnlib.util.open_url(path)
        return legacy.load_network_pkl(f)['G_ema'].to(self.device)


    def generate_image(self, z, G, psi=float):
        w = G.mapping(z, None, truncation_psi = psi, truncation_cutoff=8)
        img = G.synthesis(w, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

    def generate_image_sep(self, z, G_m, G_s, psi=float):
        w = G_m.mapping(z, None, truncation_psi = psi, truncation_cutoff=8)
        img = G_s.synthesis(w, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

    def generate_image_w(self, w, G):
        w = torch.from_numpy(w).to(self.device)
        img = G.synthesis(w, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()


    def generate_seed(self):
        z = torch.randn([1, self.G_ffhq.z_dim]).cuda()
        img = self.generate_image(z, self.G_ffhq, self.truncation_psi)
        return img, z


    def add_border(self, img):
      return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = [255, 255, 255])


    def generate_art_styles(self, seed=None):
        if seed == None:
            seed = self.generate_seed()

        #result = self.generate_image(seed, self.G_ffhq, self.truncation_psi)
        result = []
        for source in os.listdir(self.styles_dir):
            G_blend = copy.deepcopy(self.G_ffhq)
            G_new = self.generate_network(f'{self.styles_dir}/{source}')
            name = os.path.splitext(os.path.basename(source))

            mix1  = mixes.Mixes[name[0]].value
            mix2 = [1-m for m in mix1]

            mix = [mix1, mix2]

            styles = [self.G_ffhq, G_new]
            self.__mix(G_blend, styles, mix)
            img = self.generate_image(seed, G_blend, psi=self.truncation_psi)
            img = self.add_border(img)
            #result = np.hstack((result, img))
            result.append(img)
        return np.hstack(result)

    def generate_art_styles_w(self, models_source, latent_w, psi=0.5):
        w = torch.from_numpy(latent_w).to(torch.device('cuda'))
        result = self.generate_image_w(w, self.G_ffhq)
        img_arr = []
        img_arr.append(result)
        for source in os.listdir(models_source):
            G_blend = copy.deepcopy(self.G_ffhq)
            G_new = self.generate_network(f'{models_source}/{source}')
            name = os.path.splitext(os.path.basename(source))

            mix1  = mixes.Mixes[name[0]].value
            mix2 = [1-m for m in mix1]

            mix = [mix1, mix2]

            styles = [self.G_ffhq, G_new]
            self.__mix(G_blend, styles, mix)

            #add truncation
            w_avg = G_blend.mapping.w_avg
            w = torch.from_numpy(latent_w).to(torch.device('cuda'))
            w_blend = w_avg + psi*(w - w_avg)
            img = self.generate_image_w(w_blend, G_blend)
            img_arr.append(img)
            result = np.hstack((result, img))
        return result, img_arr
    

    def __mix(self, base, styles, mix):
        resolutions =  [4*2**x for x in range(int(np.log2(self.resolution)-1))]

        dest_dict = base.state_dict()
        src_dicts = [style.state_dict() for style in styles]
        w_avg = base.mapping.w_avg

        for name, _ in base.named_parameters():
            res_index = None
            for i, res in enumerate(resolutions):
                if f'synthesis.b{res}' in name:
                    res_index = i
                    break
            if res_index and not ('mapping' in name):
                w_0 = styles[0].mapping.w_avg[2*res_index]
                w_1 = styles[0].mapping.w_avg[2*res_index+1]
                w_avg = w_0*mix[0][res_index]
                w_avg += w_1*mix[0][res_index]
                data=src_dicts[0][name].data*mix[0][res_index]
                for i in range(1, len(src_dicts)):
                    w_i = styles[i].mapping.w_avg[2*res_index] 
                    w_i1 = styles[i].mapping.w_avg[2*res_index+1]
                    w_avg += w_i*mix[i][res_index]
                    w_avg += w_i1*mix[i][res_index]
                    data+=src_dicts[i][name].data*mix[i][res_index]

                dest_dict[name].data.copy_(data)

        base_dict = base.state_dict()
        base_dict.update(dest_dict) 
        base.load_state_dict(dest_dict)

    def find_latent(self, target_pil, iterations):
      w, h = target_pil.size
      s = min(w, h)
      target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
      target_pil = target_pil.resize((256, 256), PIL.Image.LANCZOS)
      target_uint8 = np.array(target_pil, dtype=np.uint8)
      target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=self.device)
      w = projector.project(self.G_ffhq, target, num_steps=iterations, device=self.device, verbose=True)
      projected_w = w[-1]
      latent = projected_w.unsqueeze(0).cpu().numpy()
      return latent



