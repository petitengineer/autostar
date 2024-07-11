#!/bin/bash

# Stop and remove the old container
docker stop flask-app || true && docker rm flask-app || true

# Rebuild the Docker image
docker build -f Dockerfile -t flask-app .

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
docker run -d -p 5000:5000 -v $(pwd):/app --network=mynetwork --name flask-app flask-app

