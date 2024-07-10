import cv2
import numpy as np
import pandas as pd

from moviepy import editor
import moviepy


#Function to find the slope and intercept of lines returned by the hough tf
def avgSlopeIntercept(lines):
    leftLines = []
    leftWeights =[]
    rightLines = []
    rightWeights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2 :
                continue

            slope = (y2-y1)/(x2-x1)

            intercept = y1 - (slope * x1)

            length = np.sqrt(((y2-y1)**2) + ((x2 - x1)**2))

            if slope < 0:
                leftLines.append((slope, intercept))
                leftWeights.append((length))
            else :
                rightLines.append((slope. intercept))
                rightWeights.append((length))

    leftLane = np.dot(leftWeights, leftLines) / np.sum(leftWeights) if len(leftWeights) > 0 else None
    rightLane = np.dot(rightWeights, rightLane) / np.sum(rightWeights) if len(rightWeights) > 0 else None

    return leftLane, rightLane


#Conver the slope and intercept of each line into pixel points
def pixel_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2-intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


#Create complete lines from pixel points
def laneLines(image, lines):
    
    leftLane, rightLane = avgSlopeIntercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6

    leftLine = pixel_points(y1, y2, leftLane)
    rightLane = pixel_points(y1, y2, rightLane)

    return leftLane, rightLane


#Draw lines on the input image
def drawLaneLines(image, lines, color = [255,0,0], thickness=12):
    
    lineImage = np.zeros_like(image)

    for line in lines:
        if line is not None:
            cv2.line(lineImage, *line, color, thickness)
    
    return cv2.addWeighted(image, 1.0, lineImage, 1.0, 0.0)

    
#Function to perform probabilistic hough transform
#Determine and cut the region of interest in the input image
def houghTransformation(image):
    #Distance resolution of the accumulator in px
    rho = 1

    #Angle resolution of the accumulator in radians
    theta = np.pi/180

    #Only lines that are greater than this threshold will be returned
    threshold = 20

    #Lines shorter than this are rejected
    minLineLength = 20
    
    #The max allowed gap between points on the same line to link them
    maxLineGap = 500

    #Return an array containing dimensions of straight lines 
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)


def regionSelection(image):
    mask = np.zeroes(image)

    #if image has more than one channel
    if len(image.shape) > 2:
        channelCount = image.shape[2]
        ignoreMaskColor = (255,) * channelCount
    
    #for single channel images
    else:
        #creating a white mask color
        ignoreMaskColor = 255

    #creating a polygon to keep focus only on the road from our input media
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    #fill this polygon with white color
    cv2.fillPoly(mask, vertices, ignoreMaskColor)

    #To get only the edges on the road, we perform bitwise AND operation
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage



def frameProcessor(image):
    #CV Pipeline
    #Image -> Grayscale -> Noise Reduction -> Edge Detection -> Mask
    #Convert RGB image to grayscale
    grayscale = cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)

    #Apply gaussian blur to remove noise
    kernelSize = 5
    blur = cv2.GaussianBlur(grayscale, (kernelSize, kernelSize), 0)

    #Applying canny edges
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    
    #Applying a mask
    region = regionSelection(edges)

    #Apply hough transform to get straight lines from the image
    hough = houghTransformation(edges)

    result = drawLaneLines(image, laneLines(image, hough))

    return result



def processVideo(inputVid, outputVid):
    #Read the input video file
    inputVideo = editor.VideoFileClip(inputVid, audio=False)

    #apply our frame processing function
    processedVideo = inputVideo.fl_image(frameProcessor)

    #save the output video as a .mp4 file
    processVideo.write_videofile(outputVid, audio=False)
    



processVideo('input.mp4','output.mp4')