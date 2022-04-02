import os
os.system("pip install gfpgan")

#os.system("pip freeze")
#os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P .")
import random
import gradio as gr
from PIL import Image
import torch
# torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1024px-Abraham_Lincoln_O-77_matte_collodion_print.jpg', 'lincoln.jpg')
# torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/5/50/Albert_Einstein_%28Nobel%29.png', 'einstein.png')
# torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Thomas_Edison2.jpg/1024px-Thomas_Edison2.jpg', 'edison.jpg')
# torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Henry_Ford_1888.jpg/1024px-Henry_Ford_1888.jpg', 'Henry.jpg')
# torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg/800px-Frida_Kahlo%2C_by_Guillermo_Kahlo.jpg', 'Frida.jpg')


import cv2
import glob
import numpy as np
from basicsr.utils import imwrite
from gfpgan import GFPGANer

bg_upsampler = None


 
# set up GFPGAN restorer
restorer = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANv1.3.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler)


def inference(img):
    input_img = cv2.imread(img, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=False, paste_back=True)
    
    #return Image.fromarray(restored_faces[0][:,:,::-1])
    return Image.fromarray(restored_img[:, :, ::-1])

title = "让美好回忆更清晰"


description = "上传老照片，点击Submit，稍等片刻，右侧Output将照片另存为即可。"

article = "<p style='text-align: center'><a href='https://huggingface.co/spaces/akhaliq/GFPGAN/' target='_blank'>clone from akhaliq@huggingface with little change</a> | <a href='https://github.com/TencentARC/GFPGAN' target='_blank'>GFPGAN Github Repo</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>"

gr.Interface(
    inference, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
    ['lincoln.jpg'],
    ['einstein.png'],
    ['edison.jpg'],
    ['Henry.jpg'],
    ['Frida.jpg']
    ]
    ).launch(enable_queue=True,cache_examples=True,share=True)
    
    
