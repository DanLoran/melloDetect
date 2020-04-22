.PHONY: all

all: docker exec

exec:
	docker run -it --rm \
	-v ${CURDIR}:/w \
	-w /w \
	mello:latest bash

docker: Dockerfile
	docker build -t mello:latest -f Dockerfile .
