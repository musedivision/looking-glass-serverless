import io
import re
import os
import sys
import json
import base64

import torch
import numpy as np
from PIL import Image

from transformer_net import TransformerNet


def convert_to_image(data):
    img = data.detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img)

def img_to_base64(img):
   outputBuffer = io.BytesIO()
   img.save(outputBuffer, format='JPEG')
   bytes64 = outputBuffer.getvalue()
   img64 = base64.b64encode(bytes64)
   return img64.decode() 

def content_transform(x):
    img = torch.from_numpy(np.array(x).transpose((2, 0, 1)))
    return img.float()

device = torch.device("cpu")

models_list = ['candy', 'mosaic', 'rain_princess', 'udnie']


def getWeights():
    # can read the weights from a url.
    # could be interesting for extending this functionto new styles
    # torch.hub.load_state_dict_from_url()	
    # setup the process
    state_dict = torch.load(os.getcwd() + "/models/mosaic.pth")

    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    
    return state_dict

class SetupModel(object):
    model = TransformerNet()

    def __init__(self, f):
        self.f = f

        self.model.load_state_dict(getWeights())
        self.model.to(device)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def predict(PIL_img):
    # print('predict')
    # input_batch = []
    # with Image.open(io.BytesIO(r)) as im:
        # im = im.convert('RGB')
        # input_batch.append(content_transform(im))

    # input_x = content_transform(PIL_img)
    img = PIL_img
    MAXWIDTH = 550
    s= img.size 
    ratio = MAXWIDTH/s[0] 
    newimg = img.resize((int(s[0]*ratio), int(s[1]*ratio)), Image.ANTIALIAS)
    # print(newimg.size)
    x = content_transform(newimg)
    x = x.unsqueeze(0).to(device)
    # print(torch.tensor([1.0]))
    # print(x.shape) 
    # x = transforms.Lambda(lambda x: x.mul(255))(x)
    # input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
    return SetupModel.model(x)
    # return False


@SetupModel  # download the model when servicing request and enable it to persist across requests in memory
def handler(event, _):
    

    imgdata = base64.decodestring(event["body"])
    # img = Image.frombytes('RGB',(100,100),base64.decodestring(imagestr))
    img = Image.open(io.BytesIO(imgdata))
    w, h = img.size
    # print(img)
    # print('asdf')
    # being paranoid and not writing user data to disc (should also be encrypted in real life)

    try:
        model_output = predict(img)
        
        img = convert_to_image(model_output[0])
        img64 = img_to_base64(img)
        # print(img64)

        return {
            'statusCode': 200,
            'body': img64
        }
        return img
    except Exception as e:
        print(str(e))
        sys.exit()

    # return model_output
    return {
        'statusCode': 200,
        'body': json.dumps(f"""height:{h} width:{w}""")
    }
