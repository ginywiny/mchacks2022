import os
import random

import numpy as np
# import subprocess
# import sys
# import io
# import shutil
# from datetime import datetime
# from flask import Flask, flash, render_template, request, redirect, url_for, session, json, jsonify, make_response,send_file,send_from_directory
from flask import Flask, render_template, Response, request, redirect, url_for, json
import cv2
from torchvision.models import resnet18
from torchvision import transforms as T
import torch
import torch.nn
import image_difference

# Session information (Cookies)
app = Flask(__name__, static_url_path='/static') #Fix to allow for local file loading??

# The internal webcam
# camera = cv2.VideoCapture(0)

# Global variable to store cart items
cartList = []
cartMap = {}
cart2List = [[]]

items_counter = 0
items = ["book", "deodorant", "protein_powder"]

# The external webcam
camera = cv2.VideoCapture("/dev/video2")
_, before_image = camera.read()

# The neural network
model = resnet18(pretrained=True)

# Homepage
# Contain video feed and shopping list
@app.route('/')
def home():
    if len(cartList) == 0:
        return render_template("home.html")
    else:
        return render_template("home.html", cartList=json.dumps(cartList))

@app.route('/list/<cartItem>')
def addItem(cartItem):
    global cartList
    if cartItem is not None:
        cartList.append(cartItem)
        cartMap[cartItem] = 1
        print(cartMap)

        randCost = random.randint(1, 20)
        cart2List.append([cartItem, randCost])
        print(cart2List)

    return render_template("home.html", cartList=json.dumps(cart2List))

# Display shopping List
@app.route('/list')
def shopping_list():
        return render_template("list.html")

# Camera image generating
def gen_camera():
    while True:
        # Read camera frames
        success, frame=camera.read()
        if (not success):
            break
        else:
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Video stream 
@app.route('/video_feed')
def video_feed():
        return Response(gen_camera(),
            mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image_capture', methods=['POST'])
def image_capture():
    # Acquire post for video frame to get item
    if request.method == "POST":
        submitPic = request.form["submitBox"]
        print("Frame submitted! " + submitPic)

        if submitPic is not None:
            item = demo(use_model=True)
            # return redirect(url_for("addItem", cartItem=submitPic))
            return redirect(url_for("addItem", cartItem=item))
        else:
            return redirect(url_for("home"))
    else:
            return redirect(url_for("home"))
        

@app.route("/testPage", methods=['POST'])
def test():
    return f"<h1>VERY GOOD</h1>"


# Clear list 
@app.route("/clear")
def clear():
    cartList.clear()
    cart2List.clear()
    return redirect(url_for("home"))

# Clear list 
@app.route("/wow")
def wow():
    return render_template("wow.html")


def demo(use_model=True):
    if use_model:
        global model
        try:
            softmax = torch.nn.Softmax(dim=1)
            image = take_picture(show_pics=True)
            tensor = T.ToTensor()(image)
            output = model(tensor.unsqueeze(0))
            output = softmax(output)
            print(f"output: {output}")
            label_index = output.argmax().numpy().item()
            return items[label_index]
        except Exception as e:
            print(f"could not predict removed object: {repr(e)}: {e.args}")

    return default()


def default():
    # so it doesn't break
    global items_counter
    item = items[items_counter%len(items)]
    items_counter += 1
    return item


def take_picture(show_pics=True):
    global camera
    _, frame = camera.read()
    image_diff = image_difference.calculate_image_difference(before_image, frame)
    bbox = image_difference.predict_removed_object(before_image, frame)
    cropped_image = image_difference.crop_bbox_from_image(bbox, before_image)
    if show_pics:
        while True:
            cv2.imshow("after", frame)
            cv2.imshow("diff", image_diff)
            cv2.imshow("cropped", cropped_image)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyWindow("after")
                cv2.destroyWindow("diff")
                cv2.destroyWindow("cropped")
                break

    return cropped_image

    
def load_model():
    global model
    model = resnet18(pretrained=True)
    num_in_features = model.fc.in_features

    # replace fully connected layer with one who's output dimension matches the number of classes
    model.fc = torch.nn.Linear(num_in_features, 3)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model.load_state_dict(torch.load(os.path.join(models_dir, "model.pth")))

    return model.eval().cpu()

if __name__ == "__main__":
    load_model()
    # Set to False to display video feed
    app.run(debug=False)
    # app.run(debug=True)
