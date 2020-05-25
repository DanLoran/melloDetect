.PHONY: all test

all: docker exec

exec:
	docker exec -it \
	mello bash

docker: build/Dockerfile
	docker build -t mello:latest -f build/Dockerfile .

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

test:
	pytest test
