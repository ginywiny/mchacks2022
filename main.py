# import os
# import subprocess
# import sys
# import io
# import shutil
# from datetime import datetime
from flask import Flask, flash, render_template, request, redirect, url_for, session, json, jsonify, make_response,send_file,send_from_directory


# Session information (Cookies)
app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static') #Fix to allow for local file loading??


# Homepage
# Contain video feed and shopping list
@app.route('/')
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
