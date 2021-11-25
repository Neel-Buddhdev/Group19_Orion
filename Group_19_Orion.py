import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


qualityOfRoad = []
qualityOfRoad.append(float(input('As per data enter no. of Vehicles Per Hour(vph) allowed on Road X : ')))
qualityOfRoad.append(float(input('As per data enter no. of Vehicles Per Hour(vph) allowed on Road Y : ')))
qualityOfRoad.append(float(input('As per data enter no. of Vehicles Per Hour(vph) allowed on Road Z : ')))
qualityOfRoad.append(float(input('As per data enter no. of Vehicles Per Hour(vph) allowed on Road W : ')))


# r --> x, i --> y
def To_RREF(Matrix):
    if not Matrix: 
        return
    counter = 0
    cnt_row = len(Matrix)
    cnt_col = len(Matrix[0])
    for x in range(cnt_row):
        if counter >= cnt_col:
            return
        y = x
        while Matrix[y][counter] == 0:
            y = y + 1
            if y == cnt_row:
                y = x
                counter = counter + 1
                if cnt_col == counter:
                    return
        Matrix[y],Matrix[x] = Matrix[x],Matrix[y]
        lv = Matrix[x][counter]
        Matrix[x] = [ mrx/float(lv) for mrx in Matrix[x]]
        for y in range(cnt_row):
            if y != x:
                lv = Matrix[y][counter]
                Matrix[y] = [ iv - lv*rv for rv,iv in zip(Matrix[x],Matrix[y])]
        counter = counter + 1


# To scan the matrix from the user

print("\nThe number of rows of the matrix : 4")
print("\nThe number of columns of the matrix : 5")
R = 4
C = 5



# User input of entries in a
# single line separated by space


print("\nEnter the entries in a single line (separated by space): \n")
entries = list(map(int, input().split()))


# For printing the matrix
matrix = np.array(entries).reshape(R, C)
print("\n\nThe matrix is given by \n", matrix)


actual_value = matrix[3][4]


#Perform row echelon reduction of the inputted matrix
A = matrix.tolist()
To_RREF(A )
A = np.array(A)


print("\n\nRow reduced echelon form of the matrix is given by \n", A)


#Slicing of the matrices
X, y = A[:, :-1], A[:, -1]


solution= np.linalg.lstsq(X, y, rcond=None)
d1 = {'Road Name':['Road X','Road Y','Road Z','Road W'],'Vehicles Allowed to Pass':qualityOfRoad,'Vehicles Passing after calculation':solution[0]}
dataFrame1 = pd.DataFrame(data=d1)


d = {'Road Name':['Road X','Road Y','Road Z','Road W','Road X','Road Y','Road Z','Road W'],'No. of Vehicles Per Hour':np.concatenate((np.array(qualityOfRoad),np.array(solution[0])),axis=None),'Road Type':['Vehicles Allowed to Pass','Vehicles Allowed to Pass','Vehicles Allowed to Pass','Vehicles Allowed to Pass','Vehicles Passing after calculation','Vehicles Passing after calculation','Vehicles Passing after calculation','Vehicles Passing after calculation']}
dataFrame = pd.DataFrame(data=d)
sns.barplot(data=dataFrame,x='Road Name',y='No. of Vehicles Per Hour',hue='Road Type')


print('\nVehicles Allowed to Pass as per data on Road X : ',qualityOfRoad[0])
print('\nVehicles Allowed to Pass as per data on Road Y: ',qualityOfRoad[1])
print('\nVehicles Allowed to Pass as per data on Road Z : ',qualityOfRoad[2])
print('\nVehicles Allowed to Pass as per data on Road W : ',qualityOfRoad[3])


print('\nVehicles Passing after calculation on Road X : ',solution[0][0])
print('\nVehicles Passing after calculation on Road Y : ',solution[0][1])
print('\nVehicles Passing after calculation on Road Z : ',solution[0][2])
print('\nVehicles Passing after calculation on Road W : ',solution[0][3])


if (qualityOfRoad[0] >= solution[0][0]):
    print('\nNo need of Traffic Flow management on Road X')
else:
    print('\nTraffic Flow Management on Road X is required')


if (qualityOfRoad[1] >= solution[0][1]):
    print('\nNo need of Traffic Flow management on Road Y')
else:
    print('\nTraffic Flow Management on Road Y is required')


if (qualityOfRoad[2] >= solution[0][2]):
    print('\nNo need of Traffic Flow management on Road Z')
else:
    print('\nTraffic Flow Management on Road Z is required')


if (qualityOfRoad[3] >= solution[0][3]):
    print('\nNo need of Traffic Flow management on Road W')
else:
    print('\nTraffic Flow Management on Road W is required')


plt.show()




'''
Scanning elements

1 1 0 0 431 1 0 0 1 255 0 0 1 1 340 0 1 1 0 516

a = [[1,1,0,0,431],
     [1,0,0,1,255],
     [0,0,1,1,340],
     [0,1,1,0,516],]    '''


