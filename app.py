from flask import Flask, render_template, request, send_from_directory
#to read and transform the image 
import cv2
import keras
from tensorflow.keras.models import Sequential
#instead of saving the model and loading the model here , we have saved the weights of the model  and used the weights to craete a new model 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np

# we have to define the whol earchitecture and load the weights for that architecture 
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.load_weights('static/model.h5')#loading weights for the model 


#wtever images we load into the page to get the output they are numnbered and stored, so that they will be used to display along wit the result.
COUNT = 0
#creating a flask webpage 
app = Flask(__name__)
#this basically removes the cache, when we upload a image to the html page, that should be cleared for the next image to be uploaded.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route('/')#if you are in local host/ then render the below html page.
def man():
    return render_template('index.html')

#upon user submitting the image, then image is stored and this home function is called
@app.route('/home', methods=['POST'])
def home():
    global COUNT
    #get the image from the HTml PAGE 
    img = request.files['image']
    
    #SAVE THE IMAGE WITH NAME count
    img.save('static/{}.jpg'.format(COUNT))   
    #read the image from the static folder    using open source Imread 
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    
     #resize the image for the model which in our case it is 32X32 
    img_arr = cv2.resize(img_arr, (128,128))
    #normalizing the image 
    img_arr = img_arr / 255.0
    #reshaping the image 128,128, because he has used the height and width as 128
    img_arr = img_arr.reshape(1, 128,128,3)
    prediction = model.predict(img_arr)
    
    #store the predictions here 
    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    # to make it ready to store the next variable 
    COUNT += 1
    #once if we have the predictions or probabilities we send this data while rendering prediction.html page  where it actually decides whether it is a cat or dog 
    return render_template('prediction.html', data=preds)

#this method sends the image that is stored in the static folder to the predictin html page 
@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



