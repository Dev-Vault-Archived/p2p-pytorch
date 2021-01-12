import os
import torch
from  torch.nn.modules.upsampling import Upsample
import numpy as np
import argparse
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from skimage.metrics import structural_similarity

def ssim(image_out, image_ref):
    image_out = np.array(image_out, dtype='float')
    image_ref = np.array(image_ref, dtype='float')

    return structural_similarity(image_out, image_ref, multichannel=True)

def mkdir(directory, mode=0o777):
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=mode)

def dir_exists(directory):
    return os.path.exists(directory)

def compress(tensor, bit):
    max_val = 2**bit - 1
    tensor = torch.clamp(tensor, 0.0, 1.0) * max_val
    tensor = torch.round(tensor)
    tensor = tensor / max_val
    return tensor

# DATASET_DIR = 'datasets'
# SRC_PATH = os.path.join(DATASET_DIR, 'src')

def crop(img_arr, block_size):
    h_b, w_b = block_size
    v_splited = np.vsplit(img_arr, img_arr.shape[0]//h_b)
    h_splited = np.concatenate([np.hsplit(col, img_arr.shape[1]//w_b) for col in v_splited], 0)
    return h_splited

def tensor2img(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.moveaxis(tensor, 0, 2)
    tensor = tensor * 255
    tensor = tensor.clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(tensor)
    return img

def generate_patches(src_path, files, set_path, crop_size, img_format, upsampling):
    img_path = os.path.join(src_path, files)
    img = Image.open(img_path).convert('RGB')

    if upsampling > 0:
        img = ToTensor()(img).unsqueeze_(0)
        m = Upsample(scale_factor=abs(upsampling), mode='nearest')
        img = m(img)
        img = tensor2img(img)

    name, _ = files.split('.')
    filedir = os.path.join(set_path, 'a')
    filedirb = os.path.join(set_path, 'b')
    if not dir_exists(filedir):
        mkdir(filedir)
        mkdir(filedirb)

    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    if crop_size == None:
        img = np.copy(img)
        img_patches = np.expand_dims(img, 0)
    else:
        rem_h = (h % crop_size[0])
        rem_w = (w % crop_size[1])
        img = img[:h-rem_h, :w-rem_w]
        img_patches = crop(img, crop_size)
    
    # print('Cropped')

    for i in range(min(len(img_patches), 5)):
        img = Image.fromarray(img_patches[i])
        # print(np.asarray(compress(torch.Tensor(img_patches[0]), 4) * (2**4 - 1)))
        imgs = tensor2img(compress(ToTensor()(img_patches[i]), 3))

        if ssim(imgs, img) > 0.75:
            # Dataset kita pilah yang memiliki hasil compressan SSIM lebih dari 0.75 maka akan di skip
            # supaya kita mendapatkan dataset yang pas semuanya tidak ngasalan
            continue

        # print('Compressed')

        img.save(
            os.path.join(filedir, '{}_{}.{}'.format(name, i, img_format))
        )
        # print('OK')
        imgs.save(
            os.path.join(filedirb, '{}_{}.{}'.format(name, i, img_format))
        )

def main(target_dataset_folder, dataset_path, bit_size, pool_size, crop_size, img_format, upsampling):
    print('[ Creating Dataset ]')
    print('Crop Size : {}'.format(crop_size))
    print('Target       : {}'.format(target_dataset_folder))
    print('Dataset       : {}'.format(dataset_path))
    print('Bit       : {}'.format(bit_size))
    print('Pool       : {}'.format(pool_size))
    print('Format    : {}'.format(img_format))

    src_path = dataset_path
    if not dir_exists(src_path):
        raise(RuntimeError('Source folder not found, please put your dataset there'))

    set_path = target_dataset_folder

    mkdir(set_path)

    img_files = os.listdir(src_path)

    bar = tqdm(img_files)
    i = 0
    max = len(bar)
    # pool = Pool(pool_size)
    for files in bar:
        generate_patches(src_path, files, set_path, crop_size, img_format, upsampling)

        bar.set_description(desc='itr: %d/%d' %(
            i, max
        ))

        i += 1
        # res = pool.apply_async(
        #     generate_patches,
        #     args=(src_path, files, set_path, crop_size, img_format, upsampling)
        # )
        # print(res)
        # break
    
    # pool.close()
    # pool.join()
    print('Dataset Created')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_dataset_folder', type=str, help='target folder where image saved')
    parser.add_argument('--dataset_path', type=str, help='target folder where image saved')
    parser.add_argument('--bit_size', type=int, help='target folder where image saved')
    parser.add_argument('--pool_size', type=int, help='target folder where image saved')
    parser.add_argument('--crop_size', type=int, help='crop size, -1 to save whole images')
    parser.add_argument('--img_format', type=str, help='image format e.g. png')
    parser.add_argument('--upsampling', type=int, default=0, help='image format e.g. png')
    
    args = parser.parse_args()

    crop_size = [args.crop_size, args.crop_size] if args.crop_size > 0 else None 
    main(args.target_dataset_folder, args.dataset_path, args.bit_size, args.pool_size, crop_size, args.img_format, args.upsampling)