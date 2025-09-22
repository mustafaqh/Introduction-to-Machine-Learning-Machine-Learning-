#exercise 4 k-NN Classification using scikit-learn
#microchips
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier


#loading data from csv file
the_data = np.genfromtxt('microchips.csv', delimiter= ',')

#separate the chips into Ok and Fail
ok_chips = []
fail_chips = []

for chip in the_data :
    if (chip[2] == 1):
        ok_chips.append(chip)
    else :
        fail_chips.append(chip)
       
# convert the tuples to NumPy array 
ok_calss = np.asarray(ok_chips)
fail_class = np.asarray(fail_chips)
        
#Plot the original microchip data using different markers for the two classes OK and Fail.
plt.figure(1)
plt.title("Original data")
plt.plot(ok_calss[:, 0], ok_calss[:, 1], 'go', markersize=3)
plt.plot(fail_class[:, 0], fail_class[:, 1], 'yo', markersize=3)


me_Step = 0.05
x_min, x_max = the_data[:, 0].min()-0.1, the_data[:, 0].max()+0.1
y_min, y_max = the_data[:, 1].min()-0.1, the_data[:, 1].max()+0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, me_Step),np.arange(y_min, y_max, me_Step)) # Mesh Grid
#nx2 matrix
xy_mesh = np.c_[xx.ravel(), yy.ravel()]

#choosing colors of the mesh plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(2)
nr = 0 #figure number
for k in [1, 3, 5, 7]:
    nr = nr + 1
    plt.subplot(2, 2, nr)     

    # Create and fit the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(the_data[:, :2], the_data[:, 2])
    
    #calculating training error and plotting the data
    tr_err = 1 - knn.score(the_data[:, :2], the_data[:, 2])
    plt.title("k={}, training errors = {:.2f}".format(k, tr_err))
    
    #classify point in the mesh
    c = knn.predict(xy_mesh)
    c = c.reshape(xx.shape)
    
    # Plot points and mesh
    plt.pcolormesh(xx, yy, c, cmap=cmap_light)
    plt.scatter(the_data[:, 0], the_data[:, 1], c=the_data[:, 2], marker='.', cmap=cmap_bold)
    
plt.show()





