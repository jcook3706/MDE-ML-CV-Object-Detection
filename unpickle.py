import pickle
import os
import glob


DR_PATH = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/mAP/input/detection-results"
READ_PATH = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/runs/predict/exp/pickles"

os.chdir(READ_PATH)
file_list = glob.glob("*.pickle")
# print(file_list)

for pickle_file in file_list:
    print(pickle_file)
    # Grab image name in order to use it to 
    image_name = pickle_file.split(".pickle",1)[0]
    image_name = image_name + ".txt"
    print(image_name)
    # Load data (deserialize)
    with open(pickle_file, 'rb') as handle:
        unserialized_data = pickle.load(handle)

    write_file = open(image_name, "x")

    for item in unserialized_data:
        bbox = item.bbox
        left = bbox.minx
        right = bbox.maxx
        top = bbox.miny
        bottom = bbox.maxy
        category = item.category.name
        confidence = item.score.value
        predict_info = category + " " + str(confidence) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n'
        write_file.write(predict_info)
