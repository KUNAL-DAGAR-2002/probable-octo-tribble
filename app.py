from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

#class names
class_names = ['Pepper_bell', 'Potato']

model = tf.keras.models.load_model("models/common.h5")




image_size = 256

def preprocessing(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path,target_size=(image_size,image_size))
    image = np.array(image)
    image = image.reshape(1,image_size,image_size,3)
    return image

def pepper_bell(img_path):
    pclass_names = ['Bacterial Spot', 'Healthy']
    model1 = tf.keras.models.load_model("models/pepperbell")
    image = preprocessing(img_path)
    a = model1.predict(image)
    return pclass_names[np.argmax(a[0])]

def potato(img_path):
    ptclass_names = ['Early Blight', 'Late Blight', 'Healthy']
    model2 = tf.keras.models.load_model("models/1")
    image = preprocessing(img_path)
    a = model2.predict(image)
    return ptclass_names[np.argmax(a[0])]

def predict_label(img_path):
    image = preprocessing(img_path)
   
    a = model.predict(image)
    temp = class_names[np.argmax(a[0])]

    if temp == "Pepper_bell":
       return pepper_bell(img_path)
    
    elif temp == "Potato":
        return potato(img_path)

def predict_class(img_path):
    image = preprocessing(img_path)
   
    a = model.predict(image)
    return class_names[np.argmax(a[0])]
    
@app.route("/")
def main():
    return render_template("home.html")

@app.route("/index")
def main1():
    return render_template("index.html")
    


@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        image = request.files['my_image']
        img_path = "static/" + image.filename
        image.save(img_path)
        p = predict_label(img_path)
        b = predict_class(img_path)
        

    return render_template("index.html", prediction = p,category = b, img_path = img_path)

if __name__ == "__main__":
    app.run(debug=True)