#exercise 1 k-NN Classification  
#microchips
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import functions as fi


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

#creats 2D NumPy array called "unknown" with shape (3,2)
unknown = np.array( [[-0.3,1.0] , [-0.5,-0.1] , [0.6,0.0]] )
plt.plot(unknown[:,0], unknown[:,1] , 'ro' , markersize=3)

#printig out the predictions for unkown microchips 
for k in [1,3,5,7] :
    print('k={}'.format(k))
    the_class = ''
    for i in range(len(unknown)) :
        s = fi.k_NN_cal(the_data,unknown[i],k,task_type = "Classification")
        if(s == 1):
            the_class = 'ok'
        else:
            the_class = 'fail'
        print("chip" + str((i + 1)) + ": " + str([round(float(val), 2) for val in unknown[i]]) + " ==> " + the_class)
        

#decision bonudary
#the size of mesh step
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
for k in [1,3,5,7] :
    nr = nr +1
    plt.subplot(2, 2, nr)     
    
    #calculating training error and plotting the data
    tr_err = fi.training_error_cal(the_data, the_data[:,2],k)
    plt.title("k={}, training errors = {}".format(k, tr_err))
    
    
    classifications = []
    #classify point in the mesh
    for x in xy_mesh :
        nearest = fi.predect(the_data,x,k,task_type="Classification")
        classifications.append([nearest])
    calasses = np.asarray(classifications)
    
    # continue with mesh shape
    clz_mesh = calasses.reshape(xx.shape)
    
    # Plot points and mesh
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
    plt.scatter(the_data[:, 0], the_data[:, 1], c=the_data[:, 2], marker='.', cmap=cmap_bold)
plt.show()