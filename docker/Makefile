SHELL := /bin/bash
# .PHONY: all clean build test mount_shell shell upload check-env

DOCKER_NAME := ripl/maps
SINGULARITY_NAME := maps.sif
DOCKER := $(shell type -p nvidia-docker || echo docker)

# HACK to use colon (https://stackoverflow.com/a/61016478/19913466)
colon := :
$(colon) := :

default: build

build:
	docker build -t $(DOCKER_NAME) -f Dockerfile ..

push:
	docker push $(DOCKER_NAME)

build_sing_from_daemon:
	singularity build $(SINGULARITY_NAME) docker-daemon$(:)//$(DOCKER_NAME)$(:)latest

build-sing: build build-sing-from-daemon
