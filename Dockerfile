# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary dependencies
RUN apt-get update && apt-get install -y \
 python3.8 \
 python3-pip \
 software-properties-common \
 git

# Add KERN Suite repository and install python3-casacore
RUN add-apt-repository -s ppa:kernsuite/kern-8 && \
 apt-add-repository multiverse && \
 apt-add-repository restricted && \
 apt-get update && \
 apt-get install -y python3-casacore

# Clone the ROBII repository
COPY . /opt/robii

# Set the working directory
WORKDIR /opt/robii

# RUN pip3 install .
RUN pip3 install .

# Set up an entrypoint to run the robii command when starting the container
ENTRYPOINT ["robii"]
