import numpy as np
import dnnlib
import torch
import copy
import cv2
from PIL import Image
import warnings

from art.projector import Projector
from art.blender import Blender
from art.align import FaceDetector
import art.legacy as legacy
import os
import urllib.request
import art.drive as drive

warnings.filterwarnings('ignore')

if not os.path.isdir('data/pickels'):
    os.mkdir('data/pickels')

if not os.path.isdir('data/pickels/metrics'):
    os.mkdir('data/pickels/metrics')

if not os.path.exists('data/pickels/metrics/vgg16.pt'):
    print('Downloading vgg16 pickel...')
    urllib.request.urlretrieve(
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt",
        'data/pickels/metrics/vgg16.pt')

if not os.path.exists('data/pickels/ffhq.pkl'):
    print('Downloading ffhq pickel...')
    urllib.request.urlretrieve(
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        'data/pickels/ffhq.pkl')

if not os.path.exists('data/pickels/metfaces.pkl'):
    print('Downloading metfaces pickel...')
    urllib.request.urlretrieve(
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
        'data/pickels/metfaces.pkl')

if not os.path.exists('data/pickels/cartoon.pkl'):
    print('Downloading cartoon pickel...')
    urllib.request.urlretrieve(
        "https://owncloud.tuwien.ac.at/index.php/s/tE509GUorHUufPg/download",
        'data/pickels/cartoon.pkl')

if not os.path.exists('data/shape_predictor_68_face_landmarks.dat'):
    print('Downloading shape predictor...')
    urllib.request.urlretrieve(
        "https://owncloud.tuwien.ac.at/index.php/s/Cq06KSypHIQEbrd/download",
        'data/shape_predictor_68_face_landmarks.dat')

device = torch.device('cuda')

base_path = 'data/pickels/ffhq.pkl'
style_paths = ['data/pickels/cartoon.pkl', 'data/pickels/metfaces.pkl']

G_kwargs = dnnlib.EasyDict()
f = dnnlib.util.open_url(base_path)
model = legacy.load_network_pkl(f, **G_kwargs)['G_ema']  # type: ignore
base = model
base.to(device)
styles = [base]

for style_path in style_paths:
    f = dnnlib.util.open_url(style_path)
    model = legacy.load_network_pkl(f, **G_kwargs)['G_ema']  # type: ignore
    model.to(device)
    styles.append(model)

class FaceWorker:
    def __init__(self):
        self.face_detector = FaceDetector()

    def extract_face(self, img_pil):
        faces = self.face_detector.extract_faces(img_pil)
        if len(faces) > 0:
            return True, faces[0]
        else:
            return False, None

    def pil_to_bytes(self, img_pil):
        _, frame = cv2.imencode('.jpg', img_pil)
        return frame.tobytes()


class ProjectorWorker:

    def __init__(self):
        self.projector = Projector(device)

    def run_projection(self, target_pil, num_steps=50):
        return self.projector.run_projection(target_pil, num_steps)


class StyleWorker:
    def __init__(self):
        self.blender = Blender(copy.deepcopy(base).to(device), styles, device)
        self.blender.generate(np.load('data/test.npy'))

    def generate(self, latent_vector, mix):
        self.blender.blend_models(mix)
        img = self.blender.generate(latent_vector)
        return img

    def generate_pil(self, latent_vector, mix):
        self.blender.blend_models(mix)
        img = self.blender.generate(latent_vector)
        return Image.fromarray(img).convert('RGB')

    def generate_bytes(self, latent_vector, mix):
        img = self.generate(latent_vector, mix)
        _, frame = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return frame.tobytes()
