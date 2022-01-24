# AutoStore

A hackathon project created for McHacks 2022.
The application is a scanless checkout system who's use case is to provide users a faster and seamless way to scan items from their cart.

## Links

1. [Youtube Video](https://devpost.com/software/autostore): 2-minute video summarizing the project and demoing the use case
2. [DevPost submission](https://devpost.com/software/autostore)

## Installation Guide

1. Create a conda environment along with the necessary python dependencies: `conda create --name mchacks -f environment.yml`
2. Activate the conda environment: `conda activate mchacks`
3. Install the pip dependencies: `pip install -r requirements.txt`

## User Guide

**NOTE:** Make sure you're using the appropriate conda environment. Don't know what that is? Then follow the installation guide first.

### Running the Scanless-Checkout System (UI, Server, Camera, ML)

1. Run `python main.py`. 

This will load the machine learning model, connect to the camera, and start the web server. This is the application's intended use case: as a scanless checkout system.

### Running the Webcam and inferring cropped image objects

1. Run `python camera.py`

This is mainly for testing the webcam, image difference detector and object class inference. It's a helpful tool for debugging.
This will open a window that streams video feed from the webcam.

Press the `ESC` key the exit the program.

Press the `spacebar` key to compute an image difference and cropping of the largest connected-component from the image difference mask.

### Training the model

1. Run `python model.py`

You need to modify code in `model.py` to make any changes to training variables such as the number of epochs, criterion function, etc.
