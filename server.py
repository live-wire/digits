from PIL import Image
from flask import Flask, request
import json
from waitress import serve
import base64
app = Flask(__name__)
from final_model import get_result
from flask import render_template

cpu = "--"

@app.route("/")
def root():
    # Loads the index.html in templates/
    return render_template('index.html', message="Hola PR!")

@app.route("/classify", methods=['GET', 'POST'])
def classify():
    content = request.get_json(silent=True)
    # Got image encoded in base 64, need to convert it to png
    blah = content['base64image']
    blah = blah.replace("data:image/png;base64,","")
    blah = blah.replace(" ","+")
    if content:
        # converting base64 image to png
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodebytes(bytes(blah,'utf-8')))
        im = Image.open("imageToSave.png")
        # now we need a jpg image from the available png
        # converting png to jpg
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, im)
        bg.save("imageToSave.jpg")
        # Getting prediction for the jpg
        result = get_result("imageToSave.jpg")
        print("PREDICTION:",result)
        return json.dumps({'Status':'OK','prediction':result})
    else:
        return json.dumps({"Status":"ERROR"})

@app.route("/job")
def job():
    return json.dumps({'Status':'OK'})

if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5001)