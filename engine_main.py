from vitgpt2imagecaptioning import image_captioning
#from dgcs_security import Security
import torch
import base64
from PIL import Image
import io
from io import BytesIO
#import cv2
import numpy as np


class EngineMain(object):
    def __init__(self, **kwargs):
        self.ENGINE = None
        self.device = 'cuda'
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
        self.feature_extractor=None
        self.tokenizer=None

    def prepare(self,**kwargs):
        try:
            if 'device' in kwargs and kwargs['device'].lower() == 'cpu':
                self.device = 'cpu'
            if 'encrypted' in kwargs and kwargs['encrypted']:
                print("Encrypted error!")
            self.ENGINE,self.feature_extractor, self.tokenizer = image_captioning()
            ## burada ENGINE dedigi model diye anlıyorum
            if self.device == 'cuda':
                self.ENGINE.cuda()
            ##self.ENGINE.eval()
        except Exception as err:
            print('Engine loading error: %s' % err)
            return err
        return 'OK'

    '''
        def forward(self, input_list):
        images = []
        for inp in input_list:
            image_data = base64.b64decode(inp)
            i_image = Image.open(BytesIO(image_data))
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.ENGINE.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        result_list = []
        for pred in preds:
            response_data = {
                'status':'SUCCESS',
                'result': pred
            }
            result_list.append(response_data)
        
        return result_list
    '''

    def forward(self, input_list):
        batch = []
        images=[]

        for inp in input_list:
            batch.append(inp['image'])
        for i in range(len(batch)):
            print("type batch[i]:",type(batch[i]))
            print("batch[i] is:",batch[i])

            image_data = base64.b64decode(batch[i])
            

            #nparr = np.frombuffer(image_data, np.uint8) #2.yol
            #i_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)#2.yol
            #i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)#2.yol

            i_image = Image.open(BytesIO(image_data)) #1.yol
            if i_image.mode != "RGB": #1.yol
                i_image = i_image.convert(mode="RGB")#1.yol

            images.append(i_image)


        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.ENGINE.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        result_list = []
        for pred in preds:
            response_data = {
                'status':'SUCCESS',
                'result': pred
            }
            result_list.append(response_data)
        
        return result_list
           