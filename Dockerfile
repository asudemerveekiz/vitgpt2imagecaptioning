
FROM python:3.10
WORKDIR /home

RUN pip3 install transformers torch Pillow 

COPY main/common .
COPY models/vitgpt2imagecaptioning . 


CMD ["sh", "dgcs_run.sh"]

