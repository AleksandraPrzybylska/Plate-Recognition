## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info

### Project Purpose
The aim of the project is to write a license plate recognition program. The photos will be taken with a smartphone or a camera with the following assumptions:
- boards in the photos will be inclined no more than ± 45 degrees from the horizontal position,
- the longer edge of the board covers over one third of the photo width,
- the angle between the optical axis of the camera and the plate plane does not exceed 45 degrees,
- ordinary plates are photographed,
- with black characters on a white background (7 characters),
- plaques will also come from outside Poznań,
- photos can be of different resolution.
	
### Requirements
- Programs should be written in Python version 3.7, using the OpenCV library.
- It is possible to use external libraries (e.g. scikit-image), but it is not allowed to use external OCR modules or ready-made, trained models that enable reading characters.
- The maximum processing time is 2s for each photo.


## Technologies
Project is created with:
* Python version: 3.7.4
* OpenCV version: 4.2.0.34
* Numpy version: 1.18.5
* Tensorflow version: 2.2.0
* Imutils version: 0.5.3

## Setup
All the necessary libraries are in the requirements file. You can install with a command:
```
$ pip install -r requirements.txt
```

To run this project, you need to pass two arguments:
 - path to images directory,
 - path to output json file

```
$ python main.py images_dir results_file_json
```
