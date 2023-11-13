from flask import Flask, render_template, Response, request, redirect
import cv2
import os

from detection_video import generate_frames, get_saved_image, get_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/detect', methods=['POST'])
def detect():
    if not request.method == "POST":
        return
    # print(request.files['video'])
    return render_template("detect_soccer.html")
    # return redirect('/')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame' )

@app.route('/image')
def image():
    PATH = r'/Users/pols/dev/cv-project/project-trials/image-search/cropped'
    filename = 'img.jpg'
    filepath = os.path.join(PATH, filename)
    if not os.path.exists(filepath):
        return Response(get_text(), mimetype='text/csv')
    return Response(get_saved_image(), mimetype='multipart/x-mixed-replace; boundary=frame' )


if __name__ == "__main__":
    app.run(debug=True, port=8081)