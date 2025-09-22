#this python file contains functions which are needed to claculate
#euclidean distance
#knn in both situations Classification and Regression
#predection for Classification task
#trainig error
#predection for Regression task


import numpy as np
import operator

def euclidean_distance(x1,y1,x2,y2) :
    d = ((x2-x1)**2) + ((y2 - y1)**2)
    distance = np.sqrt(d)
    return distance


def k_NN_cal(data_parameter1,data_parameter2,k_parameter,task_type) :
    distances = {}
    neighbors = []
    #calculating distances
    for i in range(len(data_parameter1)) :
        if task_type == "Classification" :
            the_distances = euclidean_distance(data_parameter2[0], data_parameter2[1], data_parameter1[i][0], data_parameter1[i][1])
            distances[i] = the_distances
        elif task_type == "Regression" :
            the_distances = euclidean_distance(data_parameter1[i][0], 0, data_parameter2, 0)
            # the_distances = euclidean_distance([data_parameter2[0]], 0, [data_parameter1[i][0]], 0)
            distances[i] = the_distances
        else :
            print("plaese enter valid task type : 'Classification' or 'Regression' ")
            
            
    #sortering the distances in ascending order 
    #source: https://www.askpython.com/python/dictionary/sort-a-dictionary-by-value-in-python#:~:text=itemgetter()%20method%20can%20be,m'%20values%20from%20the%20iterable. 
    #key operator has been chosed becuse it is faster when sorting a larg data base
    sorted_distances = sorted(distances.items(), key= operator.itemgetter(1)) 
    
    #collecting the nearest nighbour
    for n in range(k_parameter) :
        neighbors.append(sorted_distances[n][0])
        
    return neighbors

def predect(data_parameter1,data_parameter2,k_parameter,task_type):
    ok = 0
    fail = 0
    the_meighbors = k_NN_cal(data_parameter1,data_parameter2,k_parameter,task_type)
    for c in range(len(the_meighbors)) :
        status = data_parameter1[the_meighbors[c]][2]
        
        if (status == 1) :
            ok += 1
        else :
            fail += 1
        #deciding the status
    if (ok < fail):
        return  0
    elif (ok > fail):
        return 1
    else :
         print("error in prodecting")
            
            
#function to calculate the trainings error        
def training_error_cal(training_set,y_train,k) :
     # Initialize a counter to 0 to keep track of the number of errors made during classification.
    counter = 0 
    y_train_array = np.asarray(y_train)
    d = 0
    
    #calculate the Euclidean distance between each ex and every other ex in training set
    for x in training_set :
        dis = {}
        for ex in range(len(training_set)) :
            the_distance =   euclidean_distance(x[0], x[1], training_set[ex][0], training_set[ex][1])  
            dis[ex] = the_distance
            
        sorted_distances = sorted(dis.items(), key= operator.itemgetter(1)) 
        
        #collecting the nearest nighbour
        closest_neighbor = []
        for n in range(k) :
            closest_neighbor.append(sorted_distances[n][0])
            
        #calculating the number of ok and fail neighbours 
        ok = 0
        fail = 0
        for i in range(len(closest_neighbor)) :
            status = training_set[closest_neighbor[i]][2]
            if (status == 1) :
                ok = ok + 1
            else :
                fail = fail + 1
                
        #deciding the status
        if (ok > fail):
            status_class =  1
        else :
            status_class = 0
        
        if (y_train_array[d] != status_class):
            counter = counter + 1
                
        d = d + 1
    return counter

def reg_predect(data_parameter1,data_parameter2,k_parameter,task_type) :
    y_value = 0
    nieghbors = k_NN_cal(data_parameter1,data_parameter2,k_parameter,task_type)
    for v in range(len(nieghbors)) :
        y_value = y_value + data_parameter1[nieghbors[v]][1]
    return y_value/k_parameter
        