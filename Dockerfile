# Use the Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the environment specified in the environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Expose port 8000 to the outside world
EXPOSE 8000

# Ensure the environment is activated when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "sarfusion", "python", "main.py", "app"]
