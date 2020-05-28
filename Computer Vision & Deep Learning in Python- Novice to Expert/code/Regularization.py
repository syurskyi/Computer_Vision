# importing the required packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor



neighbors = 1
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

#Looping through L1, L2 and without regularization
for regularization in ("l1","l2",None):
    print("INFO: training the model with {} regularization".format(regularization))
    model = SGDClassifier(loss="log", penalty=regularization, max_iter=100, 
                          learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    print("INFO: evaluating the model")
    #Evaluating the accuracy and print it
    accuracy = model.score(testX, testY)
    print("INFO: evaluation accuracy the model with {} regularization is {:.2f}%".format(regularization, accuracy*100))













