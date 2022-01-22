import os
# import subprocess
# import sys
# import io
# import shutil
# from datetime import datetime
# from flask import Flask, flash, render_template, request, redirect, url_for, session, json, jsonify, make_response,send_file,send_from_directory
from flask import Flask, render_template, Response
import cv2

# Session information (Cookies)
app = Flask(__name__, static_url_path='/static') #Fix to allow for local file loading??
camera = cv2.VideoCapture(0)

# Homepage
# Contain video feed and shopping list
@app.route('/')
def home():
    return render_template("home.html")

# Display shopping List
@app.route('/list')
def list():
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


if __name__ == "__main__":
    # Set to False to display video feed
    app.run(debug=True)
