# Standard Libraries
import os
import glob as gb
import random
import re

# External Libraries
import numpy as np
import pandas as pd
import cv2
import pickle
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image, UnidentifiedImageError

# Data Science and Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Image Processing Libraries
import skimage.feature
# from skimage.feature.texture import greycomatrix, greycoprops

# Visualization and Display
from IPython.display import Image, display
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

# Deep Learning Libraries
import torch
import torchvision
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from torchvision.transforms.functional import to_pil_image
# from torchvision.transforms import functional as F
import utils
from keras.preprocessing import image as kimage
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

# Device Configuration
device = torch.device('cuda')

from PIL import Image


## HARVESTED?

def VGG16_feature_extratctor(image_path):
    model = VGG16(weights='imagenet', include_top=False)
    img = kimage.load_img(image_path, target_size=(490, 490))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    VGG16_features = model.predict(x)
    return VGG16_features

def harvest_classif_model(image_path):
    os.chdir('/home/ab')
    with open('raid/svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
    VGG16_img_features = VGG16_feature_extratctor(image_path).flatten()
    VGG16_img_features = VGG16_img_features.reshape(1, -1)  # Flatten the features
    # Predict using the trained model
    y_pred = svm_model.predict(VGG16_img_features)
    if y_pred == 0:
        print('The plant is not harvested')
    else:
        print('The plant is harvested')

    return y_pred


## PLANT SEGMENTATION

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5)) 
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)


def get_bounding_boxes_plot(image_path): 
    model_path = 'models/plant_segmentation/model17.pth'
    score_threshold=0.8
    eval_transform = get_transform(train=False)
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    image = read_image(image_path)
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
        
    # Filter predictions based on score
    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]
    # Get the number of boxes
    num_boxes = len(pred_boxes)
    print(f"Number of plants detected: {num_boxes}")
    if num_boxes == 0:
        return 0, 0

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    masks = (pred_masks > 0.7).squeeze(1)
    
    # Get the center of the image
    image_center = torch.tensor([x.shape[1] / 2, x.shape[2] / 2]).to(device)  # Add .to(device)
    # Calculate the distance of each bounding box's center to the image center
    box_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    distances = torch.norm(box_centers - image_center, dim=1)

    # Get the bounding box that is closest to the center
    closest_box = pred_boxes[distances.argmin()]

    # Draw the bounding box on the image
    output_image = draw_bounding_boxes(image, closest_box.unsqueeze(0).long(), colors="red")

    # Get the mask of the plant in the middle
    middle_mask = masks[distances.argmin()]

    # Draw the mask on the image
    output_image = draw_segmentation_masks(output_image, middle_mask.unsqueeze(0), alpha=0.5, colors="blue")

    # Plot the image
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(output_image.permute(1, 2, 0))


## SIZE PREDICTION

def get_bounding_boxes(image_path, model_path): 
    score_threshold=0.8
    eval_transform = get_transform(train=False)
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    image = read_image(image_path)
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
        
    # Filter predictions based on score
    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]
    # Get the number of boxes
    num_boxes = len(pred_boxes)
    print(f"Number of plants detected for {image_path.split('/')[-1]}: {num_boxes}")
    if num_boxes == 0:
        return 0, 0

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    masks = (pred_masks > 0.7).squeeze(1)
    # Get the center of the image
    image_center = torch.tensor([x.shape[1] / 2, x.shape[2] / 2]).to(device)  # Add .to(device)
    # Calculate the distance of each bounding box's center to the image center
    box_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    distances = torch.norm(box_centers - image_center, dim=1)

    # Get the bounding box that is closest to the center
    closest_box = pred_boxes[distances.argmin()]

    # Calculate the area of the bounding box
    #box_area = (closest_box[2] - closest_box[0]) * (closest_box[3] - closest_box[1])

    # Get the mask of the plant in the middle
    middle_mask = masks[distances.argmin()]

    # Count the number of pixels in the mask
    mask_pixels = torch.sum(middle_mask).item()

    # Calculate the normalized mask pixel count
    normalized_mask_pixel_count = mask_pixels / (x.shape[1] * x.shape[2])

  #  return box_area, normalized_mask_pixel_count
    return normalized_mask_pixel_count


def predict_size(image_path):
    regression_model = load_model('raid/regression_model3.h5')
    PATH_model = 'models/plant_segmentation/model17.pth'
    pixel_count = get_bounding_boxes(image_path, PATH_model)
    new_features = np.array([pixel_count]).reshape(-1, 1)
    predicted_value = regression_model.predict(new_features)
    print('Plant Size: ',predicted_value)
    return predicted_value


## LEAVES COUNT


def get_leaves_number(pred_boxes, bbox_data_2d):
    # Initialize the list to store the boxes that are at least 80% inside the larger box
    count_leaves = []

    # For each smaller box
    for box in pred_boxes:
        # Calculate the coordinates of the intersection box
        x1 = max(box[0], bbox_data_2d[0][0])
        y1 = max(box[1], bbox_data_2d[0][1])
        x2 = min(box[2], bbox_data_2d[0][2])
        y2 = min(box[3], bbox_data_2d[0][3])

        # Calculate the area of the intersection box
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of the smaller box
        box_area = (box[2] - box[0]) * (box[3] - box[1])

        # If the intersection area is at least 80% of the box area
        if intersection_area >= 0.9 * box_area:
            # Add the box to the list
            count_leaves.append(box)

    return len(count_leaves)


def get_central_plant(image_path, size_model_path):
    # Load the model
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(size_model_path))
    model = model.to(device) 
    model.eval()

    # Load the image
    image = F.pil_to_tensor(Image.open(image_path))

    # Convert the image to a float and normalize its values to the range [0, 1]
    image = image.float() / 255.0

    # Move the image to the same device as the model
    image = image.to(device)

    # Get the predictions
    with torch.no_grad():
        prediction = model([image])

    # Get the bounding boxes
    boxes = prediction[0]['boxes']

    # Compute the center of each bounding box
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2

    # Compute the center of the image
    image_center = torch.tensor([image.shape[-2] / 2, image.shape[-1] / 2])
    image_center = image_center.to(device)

    # Compute the distance of each bounding box center to the image center
    distances = ((centers - image_center) ** 2).sum(dim=-1).sqrt()

    # Check if distances is not empty
    if distances.numel() > 0:  
        central_box = boxes[distances.argmin()]
    else:
        central_box = torch.zeros(1, 4)  # Return a default value if distances is empty

    return central_box

def leaves_count(image_path):
    model_path = 'models/leaves_segmentation/model15.pth'
    size_model_path = 'models/plant_segmentation/model17.pth'
    num_classes = 2
    eval_transform = get_transform(train=False)

    bbox_data = get_central_plant(image_path, size_model_path)
    if torch.all(bbox_data == 0):
        # Assign a default value to bbox_data
        bbox_data = torch.zeros(1, 4)

    


    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    bbox_data_2d = bbox_data.unsqueeze(0)


    image = read_image(image_path)
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_labels = pred["labels"][keep]
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]

    num_boxes_leaves = len(pred_boxes)

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"leaves: {score:.3f}" for label, score in zip(pred_labels, pred_scores)]

    output_image = draw_bounding_boxes(image, pred_boxes.long(), pred_labels, colors="red")
    
    if bbox_data_2d.shape[1] >= 4:  # Check if bbox_data_2d has at least 4 elements in the second dimension
        output_image = draw_bounding_boxes(output_image, bbox_data_2d.long(), colors="blue")
        leaves_number = get_leaves_number(pred_boxes, bbox_data_2d)
    else:
        print(f"Warning: bbox_data_2d for image {image_path} does not have the expected shape. Skipping drawing bounding boxes.")
        leaves_number = 0
    

    masks = (pred_masks > 0.7).squeeze(1)

    colors = cm.rainbow(np.linspace(0, 1, num_boxes_leaves))
    colors = [matplotlib.colors.to_hex(color) for color in colors]

    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors=colors)
    
    #### Density calculator

    # Calculate the area of each mask by summing all the pixels in the mask
    mask_areas = pred_masks.sum(dim=[1, 2])

    # Calculate the total area of the masks
    total_mask_area = mask_areas.sum()

    # Calculate the total area of the image
    total_image_area = image.shape[1] * image.shape[2]

    # Calculate the percentage of the image area that the masks take up
    percentage = (total_mask_area / total_image_area) * 100

    density = num_boxes_leaves * percentage
    
    print(f"Number of leaves detected: {num_boxes_leaves}")
    print(f"Number of leaves inside the plant: {leaves_number}")

    print(f"The density rate of plants in the image is {density/ 10}")

    density = extract_float_value(density)

    plot_leaves(image_path)
    
    return output_image, num_boxes_leaves, leaves_number, density


## PLANT HEIGHT

def predict_height_value(image_path):
    height_regression_model = load_model('raid/height_regression_model.h5')
    new_features = VGG16_feature_extratctor(image_path).reshape(1, -1)
    predicted_value = height_regression_model.predict(new_features)
    print('Plant height: ',predicted_value)
    return predicted_value


## HASHEAD?

def get_highest_exp_file(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter the list to include only files that start with 'exp' followed by a number
    exp_files = [file for file in files if re.match(r'exp\d+', file)]

    # If there are no matching files, return None
    if not exp_files:
        return None

    # Get the numbers from the file names
    numbers = [int(re.search(r'\d+', file).group()) for file in exp_files]

    # Get the highest number
    highest_number = max(numbers)

    # Return the file with the highest number
    return f'exp{highest_number}'

'''
def check_head(image_path):
    command = ["python", "yolov5/detect.py", "--weights", "yolov5/runs/train/exp38/weights/best.pt", "--img", "256", "--conf", "0.7", "--source", image_path, "--save-txt", "--save-conf"]
    subprocess.run(command)
    exp = get_highest_exp_file('yolov5/runs/detect/')
    label_folder_path = 'yolov5/runs/detect/'+exp+'/labels/'
    txt_files = gb.glob(os.path.join(label_folder_path, '*.txt'))
    IsHead = len(txt_files) > 0
    return IsHead
'''
## HEAD SIZE


def get_bounding_boxes_headSize(image_path, model_path): 
    score_threshold=0.8
    eval_transform = get_transform(train=False)
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    image = read_image(image_path)
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
        
    # Filter predictions based on score
    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]
    # Get the number of boxes
    num_boxes = len(pred_boxes)
    print(f"Number of plants detected for {image_path.split('/')[-1]}: {num_boxes}")
    if num_boxes == 0:
        return 0, 0

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    masks = (pred_masks > 0.7).squeeze(1)
    # Get the center of the image
    image_center = torch.tensor([x.shape[1] / 2, x.shape[2] / 2]).to(device)  # Add .to(device)
    # Calculate the distance of each bounding box's center to the image center
    box_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    distances = torch.norm(box_centers - image_center, dim=1)

    # Get the bounding box that is closest to the center
    closest_box = pred_boxes[distances.argmin()]

    # Calculate the area of the bounding box
    box_area = (closest_box[2] - closest_box[0]) * (closest_box[3] - closest_box[1])

    # Get the mask of the plant in the middle
    middle_mask = masks[distances.argmin()]

    # Count the number of pixels in the mask
    mask_pixels = torch.sum(middle_mask).item()

    # Calculate the normalized mask pixel count
    normalized_mask_pixel_count = mask_pixels / (x.shape[1] * x.shape[2])

  #  return box_area, normalized_mask_pixel_count
    return normalized_mask_pixel_count, box_area


def predict_plant_size(image_path):
    PATH_model = 'models/plant_segmentation/model7.pth'
    regression_model = load_model('raid/regression_model.h5')
    pixel_count, bbox_area = get_bounding_boxes_headSize(image_path, PATH_model)
   # bbox_area = extract_float_value(bbox_area)
    new_features = np.array([pixel_count]).reshape(-1, 1)
    plant_size = regression_model.predict(new_features)
    print('Plant size: ',plant_size, ' It cover an area of ', bbox_area)
    return plant_size


def read_yolov5_labels(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def plot_yolov5_boxes(label_file, image_path):
    # Load the image
    image = mpimg.imread(image_path)

    # Create a figure and axes
    fig, axes = plt.subplots(1, 1, figsize=(14, 7))
    axes.imshow(image, origin="lower")
    axes.axis('off')

    # Print image dimensions for debugging
    print(f"Image Dimensions: {image.shape}")

    # Read YOLOv5 bounding box coordinates
    bounding_boxes = read_yolov5_labels(label_file)

    # Iterate through the boxes and plot them
    for box in bounding_boxes:
        class_id, x_center, y_center, width, height, confidence = map(float, box.split())
        class_id = int(class_id)

        # Convert from relative coordinates to absolute coordinates
        x = (x_center - width / 2) * image.shape[1]
        y = (y_center - height / 2) * image.shape[0]
        box_width = width * image.shape[1]
        box_height = height * image.shape[0]

        # Print coordinates for debugging
        print(f"Bounding Box: (x={x}, y={y}, width={box_width}, height={box_height})")

        # Create a rectangle patch
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axes
        axes.add_patch(rect)

        # Add confidence as text
    axes.text(x, y - 7, f'head: {confidence:.2f}', color='r', fontsize=11) 
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.show(block=True)


def get_central_box(label_file_path, image_path):
    bounding_boxes = read_yolov5_labels(label_file_path)
    image = mpimg.imread(image_path)
    
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    min_distance = float('inf')
    closest_box = None

    for box in bounding_boxes:
        class_id, x_center, y_center, width, height, confidence = map(float, box.split())
        class_id = int(class_id)

        # Convert from relative coordinates to absolute coordinates
        x = (x_center - width / 2) * image.shape[1]
        y = (y_center - height / 2) * image.shape[0]
        box_width = width * image.shape[1]
        box_height = height * image.shape[0]

        box_center = (x + box_width / 2, y + box_height / 2)
        distance = ((box_center[0] - image_center[0]) ** 2 + (box_center[1] - image_center[1]) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            closest_box = (x, y, box_width, box_height)

    return closest_box


def calculate_bbox_area(bbox):
    _, _, width, height = bbox
    return width * height


def get_plant_box(image_path, model_path): 
    score_threshold=0.8
    eval_transform = get_transform(train=False)
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    image = read_image(image_path)
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
        
    # Filter predictions based on score
    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]
    # Get the number of boxes
    num_boxes = len(pred_boxes)
    print(f"Number of plants detected: {num_boxes}")
    if num_boxes == 0:
        return 0, 0

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # Get the center of the image
    image_center = torch.tensor([x.shape[1] / 2, x.shape[2] / 2]).to(device)  # Add .to(device)
    # Calculate the distance of each bounding box's center to the image center
    box_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    distances = torch.norm(box_centers - image_center, dim=1)

    # Get the bounding box that is closest to the center
    closest_box = pred_boxes[distances.argmin()]

    return closest_box


def extract_float_value(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, int):
        return float(x)
    else:
        return x


def extract_float_value_toTuple(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.cpu().numpy())
    elif isinstance(x, int):
        return float(x)
    else:
        return x


def get_head_size(plant_size, plant_bbox_width, head_bbox_width):
    head_size = (head_bbox_width * plant_size) / plant_bbox_width
    print('Head size: ', head_size)
    return head_size


def is_inside(plant_box, head_box):
    # Unpack the coordinates
    plant_x1, plant_y1, plant_x2, plant_y2 = plant_box
    head_x1, head_y1, head_x2, head_y2 = head_box

    # Check if the Head Box is inside the Plant Box
    return plant_x1 <= head_x1 and plant_y1 <= head_y1 and plant_x2 >= head_x2 and plant_y2 >= head_y2


def get_head_area(image_path):   
    hasHead = check_head(image_path)

    if hasHead:
        filename =  os.path.splitext(image_path.split('/')[-1])[0]
        exp = get_highest_exp_file('yolov5/runs/detect/')
        label_file_path = 'yolov5/runs/detect/'+exp+'/labels/'+filename+'.txt'
        central_box = get_central_box(label_file_path, image_path)
        head_area = calculate_bbox_area(central_box)
        #print('Head_area: ',head_area)
        plot_yolov5_boxes(label_file_path, image_path)
        return head_area
    else:
        print('No head detected')
        return 0

def head_size(image_path, hasHead):
    os.chdir('/home/ab/')
    sizes = []
    PATH_model = 'models/plant_segmentation/model17.pth'
    plant_size = predict_plant_size(image_path)
   # head_area = get_head_area(image_path)
    if hasHead:
        filename =  os.path.splitext(image_path.split('/')[-1])[0]
        exp = get_highest_exp_file('yolov5/runs/detect/')
        label_file_path = 'yolov5/runs/detect/'+exp+'/labels/'+filename+'.txt'
        head_box = get_central_box(label_file_path, image_path)
        head_area = calculate_bbox_area(head_box)
        head_bbox_width = head_box[2]
        plant_box = get_plant_box(image_path, PATH_model)
        plant_box = extract_float_value_toTuple(plant_box)
        if plant_box == (0, 0):
            print('warning hhh')
            plant_box = (5.651826, 0.0, 256.0, 231.01167)
            
        plot_2_boxes_on_image(image_path, plant_box, head_box)
        print('4 #############################################', plant_box)
        plant_bbox_width = plant_box[2]
        print('5 #############################################', plant_bbox_width)
        print('Plant Box: ', plant_box)
        print('Head Box: ', head_box)
        if is_inside(plant_box, head_box):
            head_size = get_head_size(plant_size, plant_bbox_width, head_bbox_width)
            sizes.append(head_size)
            print('Head Size: ',sizes)
        else:
            print('################### Head is not inside plant box for ', image_path, ' ###################')
    else:
        print('################### No head detected for ', image_path, ' ###################')
    return sizes

def plot_2_boxes_on_image(image_path, plant_box, head_box):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = head_box
    x, y, w, h = int(x), int(y), int(w), int(h) 
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x, y, w, h = plant_box
    x, y, w, h = int(x), int(y), int(w), int(h)  
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def features_dataframe(isHarvested_list, predicted_size_list, predicted_height_list, hasHead_list, leaves_count_list, density_rate_list, head_size_list):
    # Unpack image_features into individual lists

    # Convert each list of tuples into a dictionary
    dicts = [dict(x) for x in [isHarvested_list, predicted_size_list, predicted_height_list, hasHead_list, leaves_count_list, density_rate_list, head_size_list]]

    # Convert each dictionary into a DataFrame
    dfs = [pd.DataFrame(list(d.items()), columns=['Date', name]) for d, name in zip(dicts, ['isHarvested', 'plant_size', 'plant_height', 'hasHead', 'leaves_count', 'density_rate', 'head_size'])]

    # Merge all the DataFrames on the 'Date' column
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on='Date')

    # Convert the 'predicted_size' and 'predicted_height' columns from numpy arrays to floats
    for column in ['plant_size', 'plant_height']:
        df[column] = df[column].apply(lambda x: x[0][0] if isinstance(x, np.ndarray) else x)

    # Set the value of 'head_size' to 0 if it has no value
    df['head_size'] = df['head_size'].apply(lambda x: 0 if isinstance(x, list) and not x else x)
    df['head_size'] = df['head_size'].apply(lambda x: x[0][0][0] if isinstance(x, list) and len(x) > 0 and len(x[0]) > 0 and len(x[0][0]) > 0 else x)
    df['head_size'] = df['head_size'].astype(float).round(2)
    # Convert the 'hasHead' column from boolean to integer
    df['hasHead'] = df['hasHead'].map({False: 0, True: 1})

    # Replace underscores with hyphens in the 'Date' column
    df['Date'] = df['Date'].str.replace('_', '-')

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Round float columns to 2 decimal places
    for column in ['plant_size', 'plant_height', 'density_rate']:
        df[column] = df[column].round(2)

    # Sort the DataFrame by the 'Date' column
    df = df.sort_values('Date')

    return df

def single_image_features_dataframe(isHarvested_list, predicted_size_list, predicted_height_list, hasHead_list, leaves_count_list, density_rate_list, head_size_list):
    # Unpack image_features into individual lists

    # Convert each list of tuples into a dictionary
    dicts = [dict(x) for x in [isHarvested_list, predicted_size_list, predicted_height_list, hasHead_list, leaves_count_list, density_rate_list, head_size_list]]

    # Convert each dictionary into a DataFrame
    dfs = [pd.DataFrame(list(d.items()), columns=['Filename', name]) for d, name in zip(dicts, ['isHarvested', 'plant_size', 'plant_height', 'hasHead', 'leaves_count', 'density_rate', 'head_size'])]

    # Merge all the DataFrames on the 'Date' column
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on='Filename')

    # Convert the 'predicted_size' and 'predicted_height' columns from numpy arrays to floats
    for column in ['plant_size', 'plant_height']:
        df[column] = df[column].apply(lambda x: x[0][0] if isinstance(x, np.ndarray) else x)

    # Set the value of 'head_size' to 0 if it has no value
    df['head_size'] = df['head_size'].apply(lambda x: 0 if isinstance(x, list) and not x else x)
    df['head_size'] = df['head_size'].apply(lambda x: x[0][0][0] if isinstance(x, list) and len(x) > 0 and len(x[0]) > 0 and len(x[0][0]) > 0 else x)
    df['head_size'] = df['head_size'].astype(float).round(2)
    # Convert the 'hasHead' column from boolean to integer
    df['hasHead'] = df['hasHead'].map({False: 0, True: 1})
    
    # Round float columns to 2 decimal places
    for column in ['plant_size', 'plant_height', 'density_rate']:
        df[column] = df[column].round(2)
    return df


##  PLOT LEAVES


def plot_leaves(image_path):
    PATH_model = 'models/leaves_segmentation/model15.pth'
    size_model_path = 'models/plant_segmentation/model10.pth'
    # Load the model
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(PATH_model))
    model = model.to(device) 
    model.eval()

    bbox_data = get_central_plant(image_path, size_model_path) 
    bbox_data_2d = bbox_data.unsqueeze(0)   
    # Read and preprocess the image
    image = read_image(image_path)
    eval_transform = get_transform(train=False)
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    # Filter predictions based on score
    pred_scores = pred["scores"]
    keep = pred_scores > 0.8
    pred_labels = pred["labels"][keep]
    pred_scores = pred_scores[keep]
    pred_boxes = pred["boxes"][keep]
    pred_masks = pred["masks"][keep]

    # Get the number of boxes
    num_boxes_leaves = len(pred_boxes)
    print(f"Number of leaves detected: {num_boxes_leaves}")

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"leaves: {score:.3f}" for label, score in zip(pred_labels, pred_scores)]

    # Draw the bounding boxes for the leaves
    output_image = draw_bounding_boxes(image, pred_boxes.long(), pred_labels, colors="red")
    # Draw the bounding box for the plant
    output_image = draw_bounding_boxes(output_image, bbox_data_2d.long(), colors="blue")
    #Get the number of leaves inside the plant
    leaves_number = get_leaves_number(pred_boxes, bbox_data_2d)
    print(f"Number of leaves inside the plant: {leaves_number}")

    masks = (pred_masks > 0.7).squeeze(1)

    # Generate a list of colors
    colors = cm.rainbow(np.linspace(0, 1, num_boxes_leaves))  # replace num_plants with num_boxes

    # Convert RGBA to hexadecimal
    colors = [matplotlib.colors.to_hex(color) for color in colors]

    # Draw the segmentation masks
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors=colors)

    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(output_image.permute(1, 2, 0))


## Features extraction funtion (SINGLE PLANT)



def print_features(single_plant_features):
    if single_plant_features['isHarvested'][0] == 1:
        print('The plant is harvested, no features to extract')
    else:
        print('The plant has a size of', single_plant_features['plant_size'][0], 'cm, and a height of', single_plant_features['plant_height'][0], 'cm. It has', single_plant_features['leaves_count'][0], 'leaves and a density rate of', single_plant_features['density_rate'][0])
        if single_plant_features['head_size'][0] != 0:
            print('The Plant has a head and it\'s size is', single_plant_features['head_size'][0], 'cm.')
    
## Features extraction funtion (MULTIPLE PLANTS)


## Plot features

def size_plot(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['plant_size'], label='Size', marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Plant Sizes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Size (cm)')
    plt.grid(True)
    plt.legend()

    plt.show()


def height_plot(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['plant_height'], label='Height', marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Heights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Height (cm)')
    plt.grid(True)
    plt.legend()

    plt.show()


def leaves_count_plot(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['leaves_count'], label='Number of Leaves', marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Number of Leaves Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Leaves')
    plt.grid(True)
    plt.legend()

    plt.show()

def density_rate_plot(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['density_rate'], label='Density Rate', marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Density Rates Over Time')
    plt.xlabel('Date')
    plt.ylabel('Density Rate')
    plt.grid(True)
    plt.legend()

    plt.show()

def head_size_plot(df):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['head_size'], label='Head Size', marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Head Sizes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Size (cm)')
    plt.grid(True)
    plt.legend()

    plt.show()

def plot_features(df, feature1, feature2):
    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df[feature1], label=feature1, marker='o', linestyle='-')
    plt.plot(df['Date'], df[feature2], label=feature2, marker='o', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title(f'{feature1} and {feature2} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()

    plt.show()

def general_plot(df):
    scaler = MinMaxScaler()

    df[['head_size', 'plant_size', 'density_rate', 'plant_height']] = scaler.fit_transform(df[['head_size', 'plant_size', 'density_rate', 'plant_height']])

    plt.figure(figsize=(10, 6))

    plt.plot(df['Date'], df['head_size'], label='Head Size', marker='o', linestyle='-')
    plt.plot(df['Date'], df['plant_size'], label='Plant Size', marker='o', linestyle='-')
    plt.plot(df['Date'], df['density_rate'], label='Density Rate', marker='o', linestyle='-')
    plt.plot(df['Date'], df['plant_height'], label='Head Height', marker='o', linestyle='-')

    # Fill the area where head_size is approximately 0
    plt.fill_between(df['Date'], df['head_size'], where=(df['head_size'] < 0.001), color='red', alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title('Features Developement Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Values')
    plt.grid(True)
    plt.legend()

    plt.show()


def get_macro_stage(single_plant_features):
    if single_plant_features['hasHead'].item() == 1:
        if single_plant_features['head_size'].item() <= 1:
            return 'PD = 41. Macro stage 4: Head development - Beginn der ,,,,head’’-Vegetationskegelbreite> 1 cm'
        elif single_plant_features['head_size'].item() <= 4.5:
            return 'PD = 43. Macro stage 4: Head development - 30% of the expected "head" diameter reached'
        elif single_plant_features['head_size'].item() <= 7.5:
            return 'PD = 45. Macro stage 4: Head development - 50% of the expected "head" diameter reached'
        elif single_plant_features['head_size'].item() <= 10.5:
            return 'PD = 47. Macro stage 4: Head development - 70% of the expected "head" diameter reached'
        else:
            return 'PD = 49. Macro stage 4: Head development - expected "head" diameter reached'
    elif single_plant_features['plant_height'].item() > 21.54:
        if single_plant_features['plant_height'].item() <= 35.9:
            return 'PD = 35. Macro stage 3: Length growth of the main shoot - 50% of the expected species/variety typical length of the main shoot reached'
        elif single_plant_features['plant_height'].item() <= 50.26:
            return 'PD = 37. Macro stage 3: Length growth of the main shoot - 70% of the expected species/variety typical length of the main shoot reached'
        else:
            return 'PD = 39. Macro stage 3: Length growth of the main shoot - expected species/variety typical length of main shoot reached'
    elif single_plant_features['leaves_count'].item() <= 9:
        l = single_plant_features['leaves_count'].item() + 10
        return f'PD = {l} .Macro stage 1: Leaf development - {single_plant_features["leaves_count"].item()} leaf unfolds'
    elif single_plant_features['isHarvested'].item() == 1:
        return 'The plant is harvested'
    else:
        return 'The plant is in an unknown stage'


def transform_dataframe(df):
    for column in ['plant_size', 'plant_height']:
        df[column] = df[column].apply(lambda x: x[0][0] if isinstance(x, np.ndarray) else x)

    df['head_size'] = df['head_size'].apply(lambda x: 0 if isinstance(x, list) and not x else x)
    df['head_size'] = df['head_size'].apply(lambda x: x[0][0][0] if isinstance(x, list) and len(x) > 0 and len(x[0]) > 0 and len(x[0][0]) > 0 else x)
    df['head_size'] = df['head_size'].astype(float).round(2)

    for column in ['plant_size', 'plant_height', 'density_rate']:
        df[column] = df[column].round(2)
    return df


def plot_feature(all_plants_dataframes, feature):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.dates as mdates

    scaler = MinMaxScaler()

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Create a new DataFrame to store all feature values
    all_feature_values = pd.DataFrame()

    # Loop over all dataframes
    for plant_name, df in all_plants_dataframes.items():
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y_%m_%d')
        # Normalize the feature column
        df[[feature]] = scaler.fit_transform(df[[feature]])
        # Plot feature over time for each plant
        plt.plot(df['date'], df[feature], label=plant_name, linestyle='--')
        # Add feature values to the new DataFrame
        all_feature_values = pd.concat([all_feature_values, df[['date', feature]]], ignore_index=True)

    # Group by date and calculate mean and standard deviation
    grouped = all_feature_values.groupby('date').agg(['mean', 'std'])

    # Plot mean and standard deviation
    plt.plot(grouped.index, grouped[(feature, 'mean')], label='Mean', color='blue', linewidth=2.0)
    plt.fill_between(grouped.index, grouped[(feature, 'mean')] - grouped[(feature, 'std')], grouped[(feature, 'mean')] + grouped[(feature, 'std')], color='blue', alpha=0.1)

    # Add labels and title
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # adjust interval for your needs
    plt.gcf().autofmt_xdate() # autoformat the x-axis date

    plt.title(f'{feature.capitalize()} over Time for Different Plants')
    plt.xlabel('Date')
    plt.ylabel(f'Normalized {feature.capitalize()}')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
