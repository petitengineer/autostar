#!/bin/bash

# Stop and remove the old container
docker stop flask-app || true && docker rm flask-app || true

# Rebuild the Docker image
docker build -t flask-app:latest .

# Run the new container
docker run -d -p 5000:5000 --name flask-app flask-app

