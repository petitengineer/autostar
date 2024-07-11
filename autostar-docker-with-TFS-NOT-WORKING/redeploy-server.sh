#!/bin/bash

# Stop and remove the old container
docker stop tensorflow-server || true && docker rm tensorflow-server || true

# Rebuild the Docker image
docker build -f Dockerfile2 -t tensorflow-server:latest .

# Check if the network exists
docker network inspect mynetwork >/dev/null 2>&1

# If the network does not exist, create it
if [ $? -ne 0 ]; then
    echo "Network 'mynetwork' does not exist. Creating..."
    docker network create mynetwork
else
    echo "Network 'mynetwork' already exists."
fi

# Run the new container
docker run -d -p 8501:8501 --network=mynetwork --name tensorflow-server tensorflow-server

