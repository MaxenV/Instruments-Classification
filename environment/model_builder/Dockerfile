# Use the official Miniconda image as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /instrument_classifier

# Install Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to the PATH
ENV PATH /opt/conda/bin:$PATH

# Create a new conda environment and install the required packages
RUN conda create -n myenv -y -c conda-forge  python=3.9.19 jupyter=1.1.1 librosa=0.10.2.post1 tensorflow-gpu=2.4.1  keras=2.4.3 python-dotenv=1.0.1 psycopg2=2.9.9 numpy=1.22.4 pandas=2.2.3 matplotlib=3.8.4 seaborn=0.13.2 ipykernel=6.29.5 &&\
    conda clean --all -f -y


# Activate the conda environment and ensure it is used with Jupyter Notebook
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
