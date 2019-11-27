SHELL:=/bin/bash


.ONESHELL:
.DEFAULT=all
.PHONY: help test



help: ## This help.
	@echo "_________________________________________________"
	@echo "XXX-     ${Proc} ${runtime}       ${uname}   -XXX"
	@echo "_________________________________________________"
	@echo "CURRENT VERSION: ${VERSION}"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

############################################################
# Commands
############################################################

compile: # compile python dependency list 
		pip-compile requirements.in > requirements.txt

deploy: # deploy with serverless framework
	serverless deploy

build: # build test container
	docker build . -t looking-glass-test

test: # run test
	@docker run --rm -v "$$PWD":/var/task --workdir=/var/task --name looking-glass-test looking-glass-test python test.py
