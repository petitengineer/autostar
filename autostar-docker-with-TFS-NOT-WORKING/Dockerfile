# Use an official Python runtime as a parent image
FROM python:3.12-slim
#FROM tensorflow/serving

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app.py
ENV PIP_ROOT_USER_ACTION=ignore
ENV FLASK_RUN_HOST=0.0.0.0

# Install pip

# Set the working directory in the container to /app
WORKDIR /app

# Copy the necessary artifacts into the image
#COPY ./artifacts /app/artifacts
# Copy only requirements.txt to the image
COPY requirements.txt .




# Add the current directory contents into the container at /app
#ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
