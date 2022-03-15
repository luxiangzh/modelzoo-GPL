ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN mkdir -p /home/app
WORKDIR /home/app
COPY ./requirements.txt /home/app

RUN apt install -y --no-install-recommends libgl1-mesa-glx
RUN pip3.7 install -r requirements.txt
