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

### Example
#### How it works?
1. Filter with canny and find contours on the image
2. Find the plate on the image by finding four vertices.
3. When you find the number plate, straighten it and find the letters.
4. Find bounding box of every letter. Reject small or too large boxes based on width and height ratio.
5. When you find letteres, sort them from left to right.
6. If it finds letter, then save it, else filter the picture and find contours.
7. Import found letters into our trained model and return result.
8. If there is no letters found, fill result with "?".

![example_car](https://user-images.githubusercontent.com/61732852/106760932-9c353f80-6634-11eb-87b3-054bd87f9a50.PNG)
![car_result](https://user-images.githubusercontent.com/61732852/106761243-e8807f80-6634-11eb-873e-4bdffdd2e51e.PNG)

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
