#!/bin/bash

# your stuff here
IMAGE_NAME=""
WANDB_KEY=""
USER_NAME=""
Dockerfile=""
GITHUB_EMAIL=""
GITHUB_USERNAME=""


helper()
{
    echo
    echo -e "$(tput bold)Helper for the docker builder$(tput sgr0)"
    echo "-----------------------------"
    echo "  -h  --help            Print this usage and exit."
    echo "  -d  --docker_file     Specify the Dockerfile to be used."
    echo "  -i  --image_name      Specify the Image name to be created."
    echo
}

# get args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            helper
            exit 0
            ;;
        -d|--docker_file)
            Dockerfile="$2"
            shift
            shift
            ;;
        -i|--image_name)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -l|--local)
            LOCALLY=true
            shift
            ;;
        *)
            echo "Argument '$key' is not defined. Terminating..."
            exit 1
            ;;
    esac
done

docker build -f "$Dockerfile" -t "$IMAGE_NAME" \
            --build-arg UID="$(id -u)" \
            --build-arg GID="$(id -g)" \
            --build-arg USER="$USER_NAME" \
            --build-arg GROUP="$(id -g -n)" \
            --build-arg WANDB_KEY="$WANDB_KEY" \
            --build-arg GITHUB_EMAIL="$GITHUB_EMAIL" \
            --build-arg GITHUB_USERNAME="$GITHUB_USERNAME" .
