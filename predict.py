import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
import random


def letsplay(n):
    out = False
    runs = 0
    while out==False and n>0:
        
        cap = cv2.VideoCapture(0)

# Category dictionary
        categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}
        
        while True:
            _, frame = cap.read()
            # Simulating mirror image
            frame = cv2.flip(frame, 1)
            
            # Got this from collect-data.py
            # Coordinates of the ROI
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            # Drawing the ROI
            # The increment/decrement by 1 is to compensate for the bounding box
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            # Extracting the ROI
            roi = frame[y1:y2, x1:x2]
            
            # Resizing the ROI so it can be fed to the model for prediction
            roi = cv2.resize(roi, (64, 64)) 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
            cv2.imshow("test", test_image)
            # Batch of 1
            result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
            prediction = {'ZERO': result[0][0], 
                          'ONE': result[0][1], 
                          'TWO': result[0][2],
                          'THREE': result[0][3],
                          'FOUR': result[0][4],
                          'FIVE': result[0][5]}
            # Sorting based on top prediction
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            # Displaying the predictions
            cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)
        
        
            
            cv2.imshow("Frame", frame)
            
            interrupt = cv2.waitKey(10)
            if interrupt & 0xFF == 27: # esc key
                break
                
         
            cap.release()
            cv2.destroyAllWindows()
        
                    
        hit = prediction[0][0]
        random_score = random.randint(1,5)
        if hit==random_score:
            out = True
            print("Sorry you got out")
        else:
            runs+=hit
        n-=1
        print("Current score is {}".format(runs))
                    
    return runs

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")
print('Welcome to hand cricket in computer version')

n = int(input('How many balls you want to play'))
runs = letsplay(n)
print("You scored: ",end='')
print(runs)
