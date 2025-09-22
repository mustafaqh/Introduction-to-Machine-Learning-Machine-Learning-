#exercise 2 k-NN Regression
#polynomial200
import numpy as np
import matplotlib.pyplot as plt
import functions as fi


#loading data from csv file
the_data = np.genfromtxt('polynomial200.csv', delimiter= ',')

#Divide the dataset into a training set of size 100, and test set of size 100.
training_data = np.array(the_data[:100])
test_data = np.array(the_data[100:])


#Plot the training and test set side-by-side in a 1×2 pattern.

plt.figure(1)
plt.subplot(1, 2, 1)
plt.title("Training set")
plt.plot(training_data[:, 0], training_data[:, 1], 'bo', markersize=3)
plt.subplot(1, 2, 2)
plt.title("Test set")
plt.plot(test_data[:, 0], test_data[:, 1], 'ro', markersize=3)

#finding the k_nn regressiion
plt.figure(2)
figure_nr = 0

for k in [1,3,5,7] :
    figure_nr = figure_nr + 1
    #dividing x_range into 100 interval
    xy= []
    for c in np.arange(0, 20, 0.2) :
       y_coordinators = fi.reg_predect(training_data,c,k,task_type = "Regression")
       xy.append([c,y_coordinators])
    reg =np.asarray(xy)    

    #calculate prediction for the training set
    pred_train = []
    for n in training_data :
        re = fi.reg_predect(training_data, n[0], k, task_type = "Regression")
        pred_train.append([n[0], re])
    training_pred = np.asarray(pred_train)
    
    #calculate prediction for the testing set    
    pred_test = []
    for n in test_data :
        re = fi.reg_predect(test_data, n[0],k,task_type = "Regression")
        pred_test.append([n[0],re])
    test_pred = np.asarray(pred_test)
    
    #calculating the MSE for both sets
    mse_training = np.mean((training_pred[:, 1] - training_data[:, 1])**2)
    mse_test = np.mean((test_pred[:, 1] - test_data[:, 1]) ** 2)
    
    #Display (plot)a 2 × 3 plot showing the k-NN regression result and the MSE training error
    plt.subplot(2,2,figure_nr)
    plt.title("k=" + str(k) + ", MSE=" + str(round(mse_training, 2)))
    plt.plot(training_data[:, 0], training_data[:, 1], 'ro', markersize=3)
    plt.plot(reg[:, 0], reg[:, 1])
    
    # present the MSE test error for k = 1, 3, 5, 7, 9, 11.
    print("MSE test error:")
    print("k=" + str(k) + ", MSE=" + str(round(mse_training, 2)))
    
#the k gives the best regression is k = 1 because it gives the smallest error
print("the k gives the best regression is k = 1")
print("because it gives the smallest error")
    
    
plt.show()