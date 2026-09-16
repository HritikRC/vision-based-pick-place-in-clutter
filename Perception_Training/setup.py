import os
import math
import random


# To modify if needed
PATH_TO_TRAINING_DATA = os.getcwd() + "/../ur5e_sdg/training_data/Camera"
WRITING_PATH = os.getcwd() + "/data"
CLASSES = ["cube", "cylinder"]
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960

def main():
    # Get the training data path
    if not os.path.exists(PATH_TO_TRAINING_DATA):
        raise Exception("Path to training data does not exist!")
    
    # Create writing directory
    if not os.path.exists(WRITING_PATH):
        os.makedirs(WRITING_PATH)
    
    # Write classes.txt file
    with open(WRITING_PATH + "/classes.txt", "w") as f:
        file_content = "".join([(i+"\n") for i in CLASSES if CLASSES.index(i) < len(CLASSES) - 1] + CLASSES[:-1])
        f.write(file_content)

    # Create YAML file
    with open(WRITING_PATH + "/dataset.yaml", "w") as f:
        writing_path_train = WRITING_PATH + "/train"
        writing_path_val = WRITING_PATH + "/val"
        file_content = "train: " + writing_path_train + "\n" + "val: " + writing_path_val + "\n\n" + "nc: " + str(len(CLASSES)) + "\n" + "names: " + str(CLASSES)
        f.write(file_content)

    # Perform train-test split
    dirs_to_create = ["/train/images", "/train/labels", "/val/images", "/val/labels"]
    for dir in dirs_to_create:
        if not os.path.exists(WRITING_PATH + dir):
            os.makedirs(WRITING_PATH + dir)

    train_test_split = 0.80 # proportion of training data
    path_to_rgb_images = PATH_TO_TRAINING_DATA + "/rgb"
    path_to_object_detections = PATH_TO_TRAINING_DATA + "/object_detection"

    list_of_rgb_files = os.listdir(path_to_rgb_images)
    number_of_images = len(list_of_rgb_files)

    count = 0
    random.shuffle(list_of_rgb_files)
    for file in list_of_rgb_files:
        file_content = ""
        # RGB Images
        with open(path_to_rgb_images + "/" + file, "rb") as f:
            file_content = f.read()
        with open(WRITING_PATH + dirs_to_create[0 if count < math.floor(train_test_split * number_of_images) else 2] + "/" + file, "wb") as f:
            f.write(file_content)
        # Object Detections
        with open(path_to_object_detections + "/" + file.split(".")[0] + ".txt", "r") as f:
            file_content = f.readlines()
        data_yolo_format = convert_KITTI_to_YOLO(file_content)
        with open(WRITING_PATH + dirs_to_create[1 if count < math.floor(train_test_split * number_of_images) else 3] + "/" + file.split(".")[0] + ".txt", "w") as f:
            f.write(data_yolo_format)        
        count += 1

def convert_KITTI_to_YOLO(text):
    data_yolo_format = ""
    for line in text:
        line_split = line.split(" ")

        # Extract class index
        class_index = str(CLASSES.index(line_split[0]))

        # KITTI format
        x_min = float(line_split[4])
        y_min = float(line_split[5])
        x_max = float(line_split[6])
        y_max = float(line_split[7])

        # YOLO format
        x_center = str(round(((x_max + x_min) / 2) / IMAGE_WIDTH, 6))
        y_center = str(round(((y_max + y_min) / 2) / IMAGE_HEIGHT, 6))
        width = str(round((x_max - x_min) / IMAGE_WIDTH, 6))
        height = str(round((y_max - y_min) / IMAGE_HEIGHT, 6))

        # Combine
        yolo_data = [class_index, x_center, y_center, width, height]
        new_yolo_line = " ".join(yolo_data)
        data_yolo_format += new_yolo_line + "\n"
        
    return data_yolo_format

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)