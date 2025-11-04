from flask import Flask, render_template, request
import os
import numpy as np
import uuid
import gdown
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "image")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


files = os.listdir('Model')

if 'my_model.keras' in files:
    pass
else:
    file_id = "10KEWuJWGVqkdJ7kyjAOAanJS7ZAedPl3"
    url =f"https://drive.google.com/uc?id={file_id}"
    output = "Model\\my_model.keras"
    gdown.download(url,output,quiet=False)

MODEL_PATH = "Model\\my_model.keras"
model = load_model(MODEL_PATH)

def check_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

@app.route('/', methods=['GET', 'POST'])
def get_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and check_file(file.filename):
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = image.load_img(filepath,target_size=(224,224))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img)

            predictions = model.predict(img)

            # Modify based on your model’s final layer
            if predictions[0][0] >0.5:
                predicted = "spatal fracture"
            else:
                predicted = "fracture dislocation"

            return render_template('index.html', label=predicted)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
