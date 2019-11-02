import numpy as np
from PIL import Image, ImageFont, ImageDraw
import json


def generate_data():
    def create_image(s):
        # s is a string of length at most 16
        image = Image.fromarray(255*np.ones((20, 135)))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('RobotoMono-Regular.ttf', 14)
        draw.text((3, 3), s, font=font)
        image_np = (np.array(image)/255).astype(int)
        return image_np

    tokens = list('abcdefghijklmnopqrstuvwxyz') + ['<START>', '<END>', ' ']
    ch_to_ix = {ch: i for i, ch in enumerate(tokens)}
    ix_to_ch = {i: ch for i, ch in enumerate(tokens)}
    images = []
    captions = []

    for _ in range(100):
        s = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=np.random.randint(1, 17)))
        image = create_image(s)
        arr = [ch_to_ix['<START>']] + [ch_to_ix[ch] for ch in s] + [ch_to_ix['<END>']]
        arr += [ch_to_ix[' ']]*(16-len(s)) # add padding
        images.append(image)
        captions.append(arr)
        
    images = np.array(images)
    np.save('images.npy', images)
    with open('captions.json', 'w+') as f:
        json.dump(captions, f)