#import the required packages
import cv2
import numpy as np
import os

class DsLoader:
    def __init__(self, preprocessors=None):
        #save the preprocessors passed in (if any)
        self.preprocessors = preprocessors
        
        #initialize an empty list if the passed preprocessors is empty
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths):
        #initialize the lists for data and labels
        data = []
        labels = []
        
        #loop through all the image paths passed in
        for(i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            #eg path \code\datasets\animals\cats\cat1.jpg
            # we take the label name as 'cats'
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # displaying the progress in format 
            # Processed 500/3000
            if i > 500 and (i+1)%500 ==0:
                print("Processed {}/{}".format(i+1, len(imagePaths)))
                
            data.append(image)
            labels.append(label)
            
        #return the tuple with two arrays, data and labels to the code that called this function    
        return(np.array(data), np.array(labels))  
            
            
            
            
            
            
            
            
            
            
            
       