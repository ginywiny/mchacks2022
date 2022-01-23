import os
import random
# import subprocess
# import sys
# import io
# import shutil
# from datetime import datetime
# from flask import Flask, flash, render_template, request, redirect, url_for, session, json, jsonify, make_response,send_file,send_from_directory
from flask import Flask, render_template, Response, request, redirect, url_for, json
import cv2

# Session information (Cookies)
app = Flask(__name__, static_url_path='/static') #Fix to allow for local file loading??
camera = cv2.VideoCapture(0)

# Global variable to store cart items
cartList = []
cartMap = {}
cart2List = [[]]

items_counter = 0
items = ["book", "deodorant", "protein_powder"]

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
            item = demo()
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


def demo():
    global items_counter
    item = items[items_counter%len(items)]
    items_counter += 1
    return item


if __name__ == "__main__":
    # Set to False to display video feed
    # app.run(debug=False)
    app.run(debug=True)
