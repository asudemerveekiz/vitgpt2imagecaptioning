from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import base64
#import cv2
from PIL import Image
from io import BytesIO
import os
import numpy as np
app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Gelen JSON içeriğini al
        json_data = request.get_json()

        # Gelen base64 encoded resmi işle
        image_base64 = json_data.get('image')
        result = predict_step([image_base64])

        # Güncelleme: Status ekleniyor
        response_data = {
            'status': 'SUCCESS', 
            'result': result
        }
        return jsonify(response_data)
    
    except Exception as e:
        # Güncelleme: Status ekleniyor
        error_data = {
            'status': 'ERROR',
            'error_message': str(e)
        }
        return jsonify(error_data)
    

def predict_step(images_base64):
    images = []
    for image_base64 in images_base64:
        image_data = base64.b64decode(image_base64)

        #nparr = np.frombuffer(image_data, np.uint8)
        #i_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
        i_image = Image.open(BytesIO(image_data))
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
