
import time
import cv2
import numpy as np



def estimated_time(completed,num_episodes):
    estimated = ((100 - completed )/100)*num_episodes*(end-start)
    hour = estimated // 3600
    minutes = (estimated % 3600) // 60
    sek = (estimated % 3600) % 60

    return hour,minutes,sek


def state_simplify(state):
    #Remove different grass colors
    for i,x in enumerate(state):
        for j,y in enumerate(x):
                if y[0] == 102 and y[1] == 229 and y[2]== 102:
                        state[i,j] = [102,204,102]
                
                if (y[0] == 255 and y[1] == 255 and y[2]== 255) or (y[0] == 255 and y[1] == 0 and y[2]== 0):
                        state[i,j] = [102,102,102]
    return state

def rgb_to_grayscale(image):
    #convert image from RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype("float32")/255
    
    return gray



def state_contrast_add(state):
    #Remove different grass colors
    for i,x in enumerate(state):
        for j,y in enumerate(x):
                if y == np.float32(107/255) or y == np.float32(102/255)or y == np.float32(105/255):
                    state[i,j] = 1
                if y == np.float32(60/255):
                    state[i,j] = 0
    return state



def state_preproces(state):
    return state_contrast_add(rgb_to_grayscale(state_simplify(state)))
