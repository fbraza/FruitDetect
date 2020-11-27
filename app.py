from flask import Flask, redirect, url_for, render_template, request, Response
from lib_prediction.camera import Camera
from flask_socketio import SocketIO

# Camera IDs
ID_CAMERA_THUMB = 0
ID_CAMERA_FRUIT = 1

# Number of Frames for Prediction
NB_FRAMES_THUMB_PREDICTION = 30
NB_FRAMES_FRUIT_PREDICTION = 40

app = Flask(__name__)
socketio = SocketIO(app, async_mode=None)
camera_thumb = Camera(socketio, video_source=ID_CAMERA_THUMB,
                      nb_predicted_images=NB_FRAMES_THUMB_PREDICTION)
camera_fruit = Camera(socketio, video_source=ID_CAMERA_FRUIT,
                      nb_predicted_images=NB_FRAMES_FRUIT_PREDICTION)
fruit_prediction = None


def gen_thumb(camera):
    """
    Function defined to create a generator for the thumb detection.
    This generator get a frame( image) from the camera and provide
    a specific structure (in the yield command) that can be send to
    the web page

    Args:
    -----
    - camera : instance of the class Camera

    Returns:
    --------
    - specific structure return by the yield function including the frame

    """
    while True:
        frame = camera.get_frame_thumb()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


def gen_fruit(camera):
    """
    Function defined to create a generator for the fruit detection.
    This generator get a frame (image) from the camera and provide
    a specific structure (in the yield command) that can be send to
    the web page

    Args:
    -----
    - camera : instance of the class Camera

    Returns:
    --------
    - None

    """
    while True:
        frame = camera.get_frame_fruit()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["POST", "GET"])
def index():
    """
    This function treats the behavior of the index.html page

    Args:
    -----
    - None

    Returns:
    --------
    - GET: return the template index.html
    - POST: redicrect to the route /fruit_video
    """
    global fruit_prediction
    if request.method == "POST":
        return redirect(url_for("fruit_video"))
    else:
        fruit_prediction = None
        return render_template("index.html")


@app.route('/thumb_video/<string:predict>')
def thumb_video(predict):
    """
    This function treat the behavior of the thumb_video page.
    When the page is rendered, the camera is started to realize
    the prediction the code <img src="{{ url_for('thumb_video_feed') }}">
    in the thumb_video.html file "call" the route /thumb_video_feed in
    order to get an image

    Args:
    -----
    - predict: name of the predicted fruit in a string

    Returns:
    --------
    - GET: return the template thumb_video.html. Display which fruit has been
    detected and start the thumb detection
    """
    global fruit_prediction
    fruit_prediction = predict
    camera_thumb.run()
    return render_template('thumb_video.html', predict=fruit_prediction)


@app.route("/thumb_video_feed")
def thumb_video_feed():
    """
    This function  delivers thumb images to the thumb_video.html
    web page using an understandable structure

    Returns:
    --------
    - a "Response" that can be displayed as an image by the web page.
    """
    return Response(gen_thumb(camera_thumb),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/fruit_video_feed")
def fruit_video_feed():
    """
    This function delivers thumb images to the fruit_video.html
    web page using an understandable structure

    Returns:
    --------
    - a "Response" that can be displayed as an image by the web page.
    Using the generator to get images from the camera
    """
    return Response(gen_fruit(camera_fruit),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/fruit_video/')
def fruit_video():
    """
    This function treats the behavior of the fruit_video page when the page
    is rendered, the camera is started to realize the prediction the code
    <img src="{{ url_for('fruit_video_feed') }}"> in the fruit_video.html file
    "call" the route /thumb_video_feed in order to get an image.

    Returns:
    --------
    - GET: return the template fruit_video.html .
    """
    camera_fruit.run()
    return render_template('fruit_video.html')


@app.route('/fruit_manual_choice/', methods=["POST", "GET"])
def fruit_manual_choice():
    """
    This function treats the behavior of the fruit_manual_choice page

    Returns:
    --------
    - GET: return the template fruit_manual_choice.html
    - POST: Treat the answer to the manual validation of the fruit.
    go back to index page if the answer is no print the ticket if the
    answer is yes
    """
    global fruit_prediction
    if request.method == 'POST':
        if request.form['yesno'] == 'Yes':
            return redirect(url_for("ticket_printing"))
        elif request.form['yesno'] == 'No':
            return redirect(url_for("index"))
        else:
            pass
    else:
        return render_template('fruit_manual_choice.html',
                               predict=fruit_prediction)


@app.route('/ticket_printing/')
def ticket_printing():
    """
    This function treat the behavior of the ticket_print page
    It gets the fruit_detection from the global variable
    fruit_detection and send it to the ticket_printitng.html
    page to be displayed

    Returns:
    --------
    - GET: return the template ticket_printing.html.
    """
    global fruit_prediction
    return render_template('ticket_printing.html', predict=fruit_prediction)


@app.route('/under_the_hood/')
def under_the_hood():
    """
    This function render the web page under_the_hood

    Returns:
    --------
    - GET: return the template under_the_hood.html.
    """
    return render_template('under_the_hood.html')


@app.route('/authors/')
def authors():
    """
    This function render the web page authors

    Returns:
    --------
    - GET: return the template authors.html.
    """
    return render_template('authors.html')


@app.route('/thanks/')
def thanks():
    """
    This function render the web page thanks

    Returns:
    --------
    - GET: return the template thanks.html.
    """
    return render_template('thanks.html')


@app.route('/quel_fruit/')
def quel_fruit():
    """
    This function render the web page quel_fruit

    Returns:
    --------
    - GET: return the template quel_fruit.html.
    """
    return render_template('quel_fruit.html')


if __name__ == "__main__":
    print("[INFO] Starting severt at http://localhost:5001")
    socketio.run(app=app, port=5001)
