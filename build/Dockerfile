FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y \
	python3 \
	python3-pip

RUN pip3 install \
	pillow==7.1.1 \
	sklearn \
	torchvision==0.6.0 \
	argparse==1.4.0 \
	visdom==0.1.8.9 \
	tqdm==4.45.0
