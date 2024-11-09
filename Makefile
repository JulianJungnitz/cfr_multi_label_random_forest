
### PLEASE ADJUST THIS TO MATCH YOUR DEFINED APP NAME
DOCKER_IMAGE_NAME=cfr_multi_label_random_forest


### All these Values should be correctly preconfigured
REGISTRY=featurecloud.ai
DOCKER_IMAGE_VERSION=latest
DOCKER_IMAGE_TAGNAME=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

default: build ## default = build

build: ## build the image
	docker build -t $(DOCKER_IMAGE_TAGNAME) .
	docker tag $(DOCKER_IMAGE_TAGNAME) $(REGISTRY)/$(DOCKER_IMAGE_TAGNAME)

push: ## push image to docker registry
	docker push $(REGISTRY)/$(DOCKER_IMAGE_NAME):latest

test: ## test the image
	docker run --rm $(DOCKER_IMAGE_TAGNAME) /bin/echo "Success."

build_and_start_docker:
	docker kill $(DOCKER_IMAGE_NAME) || true
	docker rmi -f $(DOCKER_IMAGE_TAGNAME)
	docker build -t $(DOCKER_IMAGE_TAGNAME) .
	docker tag $(DOCKER_IMAGE_TAGNAME) $(REGISTRY)/$(DOCKER_IMAGE_TAGNAME)
	docker run -d --rm --name $(DOCKER_IMAGE_NAME) -v ./config.yml:/mnt/input/config.yml -v ./data/output:/mnt/output -p 9000:9000 featurecloud.ai/$(DOCKER_IMAGE_NAME):latest

start_task:
	curl --location 'http://localhost:9000/setup' --header 'Content-Type: application/json' --data '{"id": "0000000000000000","coordinator": false,"coordinatorID": "0000000000000000","clients": []}'
	docker logs -f $(DOCKER_IMAGE_NAME)


rmi: ## remove the image
	docker rmi -f $(DOCKER_IMAGE_TAGNAME)

rebuild: rmi build ## rebuild it

run: ## run the image
	docker run $(DOCKER_IMAGE_NAME):latest

help: ## This help dialog
	@IFS=$$'\n' ; \
		help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//'`); \
		for help_line in $${help_lines[@]}; do \
			IFS=$$'#' ; \
			help_split=($$help_line) ; \
			help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
			help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
			printf "%-10s %s\n" $$help_command $$help_info ; \
		done

.PHONY: help run list build test push rmi rebuild