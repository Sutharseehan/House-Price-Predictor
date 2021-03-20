#  Description: This program predicts if the price of a house will be above the median price or not
#               based off of it's features using deep learning

# Import the dependencies

# Intialize Artificial Neural Network
import pandas as pd
from keras.models import Sequential

# Tell us the number of layers, neurons per layer and activation function
from keras.layers import Dense

# To split the data into training and testing sets
from sklearn.model_selection import train_test_split

# To scale the data
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
fig = plt.figure()

# Load the data using pandas
df = pd.read_csv(
    '/Users/sutharseehannavajeevayokan/Desktop/Github Projects/house_price_prediction/housepricedata.csv')

# Print the first 7 rows of the data
print(df.head(7))

# Convert data into an array
dataset = df.values
print(dataset)

# Split the data into independent and dependent data sets
X = dataset[:, 0:10]  # Get all the rows from columns [0, 10)
# Get all the rows from column at position 10 (the 11th column)
Y = dataset[:, 10]

# Use the min-max scalar method from preprocessing which scales the data set so that all the features lie between 0 and 1 inclusive
min_max_scalar = MinMaxScaler()
X_scale = min_max_scalar.fit_transform(X)
print("Printing X_Scale .............")
print(X_scale)

# Split the data into 80% training and 20% (testing(10%) and validating(10%))
#                                                                  (independent, target, test size)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(
    X_scale, Y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_val_and_test, Y_val_and_test, test_size=0.5)
# The training set has 1168 data points while the vaildation and test set have 146 data points each.
# The X variables have 10 input features
print(X_train.shape, X_val.shape, X_test.shape,
      Y_train.shape, Y_val.shape, Y_test.shape)


# Build the model and architecture of the deep neural network
model = Sequential()  # initialize the Artifical Neural Network
model.add(Dense(units=32, activation="relu", input_dim=10))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

# Loss function measures how well the model did on training, and then tries to improve on it using the optimizer
model.compile(optimizer="sgd", loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model
hist = model.fit(
    X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val)
)


print(model.evaluate(X_test, Y_test)[1])

# Make a prediction
prediction = model.predict(X_test)
prediction = [1 if y >= 0.87 else 0 for y in prediction]
print("Prediction: ", prediction)
print(Y_test)

# Visualize the training loss and the validation loss to see if the model is overfitting
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc="upper right")
print(plt)
fig.savefig('Loss.png')

# Visualize the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc="lower right")
print(plt)
fig.savefig('Accuracy.png')
