import os
import torch
import cv2
import numpy as np


def generate_frames():
    # THRESHOLD = 0.75
    COUNT = 0
    # PATH = r'/Users/pols/dev/cv-project/project-trials/yolo-pytorch/cropped'
    classes = [0]

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.classes = [26, 32, 34]  # handbag, football- 32

    video_file = r'/Users/pols/dev/cv-project/videos for cv/mixkit-footballer-headbutting-the-ball-2923-medium.mp4'
    # video_file = r'/Users/pols/dev/cv-project/videos for cv/mixkit-woman-taking-pictures-out-of-a-yellow-bag-44124-medium.mp4'
    # video_file = r'/Users/pols/dev/cv-project/videos for cv/mixkit-female-photographer-taking-product-photos-on-a-set-44119-medium.mp4'
    # video_file = r'/Users/pols/dev/cv-project/videos for cv/video_preview_h264.mp4'
    # video_file = r'/Users/pols/dev/cv-project/videos for cv/mixkit-baseball-batter-871-medium.mp4'
    cap = cv2.VideoCapture(video_file)

    # num_frames = 1
    while True:
        ret, frame = cap.read()
        results = model(frame)
        save_image(results, frame.copy())
        bbframe = np.squeeze(results.render())
        ret, buffer = cv2.imencode('.jpg', bbframe)
        byte_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame +
               b'\r\n')


def save_image(results, img):
    THRESHOLD = 0.2
    PATH = r'/Users/pols/dev/cv-project/project-trials/image-search/cropped'

    df = results.pandas().xyxy[0]
    for ind in df.index:
        print(df['confidence'][ind], df['name'][ind], df['class'][ind])
        conf = df['confidence'][ind]
        objclass = name = df['class'][ind]
        name = df['name'][ind]
        if conf > THRESHOLD:
            xmin = int(df['xmin'][ind])
            xmax = int(df['xmax'][ind])
            ymin = int(df['ymin'][ind])
            ymax = int(df['ymax'][ind])

            cropped = img[ymin:ymax, xmin:xmax]
            # filepath = os.path.join(PATH, name)
            # if not os.path.exists(filepath):
            # os.makedirs(filepath)
            # print(f"{name} directory created")
            # filename = str(COUNT) + '_' + str(objclass) + '_' + str(name) + '.jpg'
            # i = COUNT
            # filename = 'img_%03d.jpg' % (COUNT,)
            filename = 'img.jpg'
            # 'data_%d.dat'%(i,)
            cv2.imwrite(os.path.join(PATH, filename), cropped)
            # COUNT += 1


def get_text():
    while True:
        yield "Hello"


def get_saved_image():
    # print("Getting saved image")
    while True:
        # print("Getting saved image")
        PATH = r'/Users/pols/dev/cv-project/project-trials/image-search/cropped'
        filename = 'img.jpg'

        filepath = os.path.join(PATH, filename)
        # if os.path.exists(filepath):
        # print("in if")
        img = cv2.imread(os.path.join(PATH, filename))
        ret, buffer = cv2.imencode('.jpg', img)
        byte_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame +
               b'\r\n')
        # else:
        #     print("in else")
        #     yield "Hello"
