import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}

    t1_start = perf_counter()
    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image)
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:",t1_stop - t1_start)
    # print("liczba zdjec:", i)
    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()
