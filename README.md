# Instruments-Classification

# Requirements
1. The project for optimal performance should be run on Linux.
2. The project requires the installation of the docker environment and the NVIDIA Container Toolkit driver in a version compatible with the version of the graphics card driver installed on the system.

# How to start
1. For training, download the dataset from the link: https://zenodo.org/records/4588740#.YFDoDdyCFPY and save it in a folder named `good-sounds` in the root directory of the project.
2. Run the `docker compose up` command from the `environment` directory. Docker will create and run the appropriate containers.
3. The running application runs on port 8000 in turn the jupyter notebook server runs on port 8888.

# Model Tests
## Model training 
![image](https://github.com/user-attachments/assets/11cb1d1e-ffe4-4da6-a764-dda2b0d74918)
## Cross Validation
![image](https://github.com/user-attachments/assets/17c41f76-e7c6-4fcc-8ea8-e14afaf231c5)
## Scikit-learn metrics
![image](https://github.com/user-attachments/assets/10680304-3c0e-4d1f-925d-9ba190e79fc6)
