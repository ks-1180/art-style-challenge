import torch
import dnnlib
import legacy
import os
import pandas as pd
import numpy as np
import cv2
import copy

def test_fid(csv_source, art_style):
    table = pd.read_csv(csv_source)
    table['ticks'] = [tick*5 for tick in range(len(table))]
    table['name'] = art_style
    table.head()
    table.loc[table.groupby('name')['fid'].idxmin()]

class Evaluator:
    def __init__(self, art_style, training_runs, idx_best_model):
        self.art_style = art_style
        self.resolution = 256
        self.base_model = f'ffhq{self.resolution}.pkl'

        self.best_FID = idx_best_model

        self.snapshot_paths = [f'{training_runs}/{p}' for p in os.listdir(f'{training_runs}/') if 'snapshot' in p]
        self.snapshot_paths.sort()

        self.G_ffhq = self.generate_network(f'/content/drive/MyDrive/data/nvidia-ada-models/{self.base_model}')
        self.G_best = self.generate_network(self.snapshot_paths[self.best_FID])

        self.seeds = [torch.randn([1, self.G_ffhq.z_dim]).cuda() for i in range(6)]
        self.truncation_psi = 0.5

    def generate_network(self, path):
        device = torch.device('cuda')
        f = dnnlib.util.open_url(path)
        return legacy.load_network_pkl(f)['G_ema'].to(device)

    def generate_image(self, z, G, psi=float):
        w = G.mapping(z, None, truncation_psi = psi, truncation_cutoff=8)
        img = G.synthesis(w, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

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
        self.base.load_state_dict(dest_dict)

    def compare_truncation(self, psi_1, psi_2):
        z = torch.randn([1, self.G_best.z_dim]).cuda()

        img1 = self.generate_image(z, self.G_best, psi=psi_1)
        img2 = self.generate_image(z, self.G_best, psi=psi_2)

        img = np.hstack((img1, img2))

        return img

    def training_stack(self):
        original_img = [self.generate_image(z, self.G_ffhq, 1.0) for z in self.seeds]
        img_rows = []
        img_rows.append(np.hstack((img for img in original_img)))

        for p in self.snapshot_paths:
            G_new = self.generate_network(p)
            img_row = []
            for seed in self.seeds:
                img = self.generate_image(self.seeds, G_new, self.truncation_psi)
                img_row.append(img)
            img_rows.append(np.hstack((i for i in img_row)))

        return np.vstack((img for img in img_rows))

    def make_video(self, outdir):
        seeds_video = self.seeds.copy()

        num_layers = int(np.log2(self.resolution)-1)
        steps = [i/10 for i in range(0,10)]

        out_dir = f'{outdir}/training.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_dir, fourcc, 15, (self.resolution*len(seeds_video), self.resolution))

        G_blend = copy.deepcopy(self.G_ffhq)
        G = self.G_ffhq

        for p in self.snapshot_paths:
            G_new = self.generate_network(p)
            styles = [G, G_new]

            for step in steps:
                mix1 = [1-step for i in range(num_layers)]
                mix2 = [step for i in range(num_layers)]

                self.__mix(G_blend, styles,[mix1, mix2])
                img_row = []
                for s in self.seeds:
                    #w = G.mapping(s, None, truncation_psi=self.truncation_psi, truncation_cutoff=8)
                    #w_new = G_new.mapping(s, None, truncation_psi=self.truncation_psi, truncation_cutoff=8)
                    # blend weight vectors
                    #w_mix = w*(1-step) + w_new*(step)
                    #img = model_mixer.generate(w_mix.cpu().detach().numpy())

                    img = self.generate_image(s, G_blend, psi=self.truncation_psi)
                    img_row.append(img)

                frame = np.hstack((i for i in img_row))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            G = G_new
        writer.release()

    def generate_mixed_styles(self, mix1, mix2):
        G_blend = copy.deepcopy(self.G_ffhq)

        styles = [self.G_ffhq, self.G_best]

        mix1 = [0, 0.2, 0.0, 0.2, 0.1, 0.2, 0.2]
        mix2 = [1-m for m in mix1]

        mix_list = [mix1, mix2]

        self.__mix(G_blend, styles, mix_list)

        results = []
        for seed in self.seeds:
            img1 = self.generate_image(seed, self.G_ffhq, psi=self.truncation_psi)
            img2 = self.generate_image(seed, G_blend, psi=self.truncation_psi)
            img = np.vstack((img1, img2))
            results.append(img)

        img_stack = np.hstack((i for i in results))

        return img_stack

    


