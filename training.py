import numpy as np
import glob
import cv2


training_data = np.zeros((1, 62500))
labels = np.zeros((1, 4), 'float')
train = glob.glob('training_data/*.npz')
#extracting data from the saved .npz files
for i in train:
    with np.load(i) as data:
        print data.files
        training_temp = data['training_image_array']
        labels_temp = data['output_array']
    training_data = np.vstack((training_data, training_temp))
    labels = np.vstack((labels, labels_temp))


training_data = training_data[1:, :]
labels = labels[1:, :]

print training_data.shape
print labels.shape

e1 = cv2.getTickCount()

layer_size = np.int32([62500, 32, 16, 8, 4])
#Creating MultiLayer Perceptrons.
neural = cv2.ANN_MLP()
neural.create(layer_size)

criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
params = dict(term_crit = criteria,
                  train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                  bp_dw_scale = 0.001,
                  bp_moment_scale = 0.0)
print "Training MLP............."
iterations = neural.train(training_data, labels, None, params = params)

e2 = cv2.getTickCount()
time_taken = (e2-e1)/cv2.getTickFrequency()
print "Time taken to train : ", time_taken

#saving the parameters
neural.save('testing.xml')

print "Number of iterations : ", iterations

ret, resp = neural.predict(training_data)
predict = resp.argmax(-1)
print "Prediction : ", predict
true_labels = labels.argmax(-1)
print "True labels : ", true_labels

print "Testing........"
train_rate = np.mean(predict == true_labels)
print "Train rate = ", (train_rate*100)