.PHONY: all

all: docker exec

exec:
	docker exec -it \
	mello bash

docker: Dockerfile
	docker build -t mello:latest -f Dockerfile .

run:
	docker run -d --rm \
	-v ${CURDIR}:/w \
	-w /w \
	--name mello \
	-p 8097:8097 \
	mello:latest \
	python3 -m visdom.server

stop:
	docker stop mello
