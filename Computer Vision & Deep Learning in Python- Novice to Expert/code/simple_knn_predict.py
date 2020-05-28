# importing the required packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
from keras.preprocessing.image import load_img, img_to_array


neighbors = 3
# number of neighbors for k-NN 
jobs = -1
# number of jobs for k-NN distance

# get the list of images from the dataset path
image_paths = list(paths.list_images('datasets/animals'))

print("INFO: loading and preprocessing")
#loading and preprocessing images using the classes created
# create instances for the loader and preprocessor classes
dp = DsPreprocessor(32, 32)
dl = DsLoader(preprocessors=[dp])
(data, labels) = dl.load(image_paths)

# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))
print("INFO: Memory size of feature matrix {:.1f}MB".format(data.nbytes/(1024*1000.0)))

# Encode the string labels as integers like 0,1,2..
le = LabelEncoder()
labels = le.fit_transform(labels)

print("INFO: splitting the dataset")
# split 25 percentage for testing and rest for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=40)

print("INFO: training the model")
#training the KNN classifier using he 75 percent of training data
model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
model.fit(trainX, trainY)

print("INFO: evaluating the model")
#Evaluating the printing the report based on test data classification
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

#a simple array of class names in animals dataset
animals_classes = ['cat', 'dog', 'panda']
#loading the unkown image for prediction
unknown_image = load_img('images/test3.jpg')
#resize the image as 32x32 pixels
unknown_image = unknown_image.resize((32,32))
#convert the resized image to array
unknown_image_array = img_to_array(unknown_image)
#reshaping the array to one row and unknown columns
unknown_image_array = unknown_image_array.reshape((1, -1))
#unknown_image_array = unknown_image_array.reshape((1, 3072))

#do the prediction using model
prediction = model.predict(unknown_image_array)
print("The predicted animal is ")
#print the corresponding label from the array animals_classes
print(str([animals_classes[int(prediction)]]))







