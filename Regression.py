import numpy as np
import matplotlib.pyplot as plt
# Take in a set of sample data and produce the corresponding linear regression coefficients
# Input data - matrix / 2D array
# where first m columns are independent variables at each sample point
# and last column is the dependent variable (or some format of this sort that works for you)
# Calculate and return the correlation coefficient.
def linearregression(matrix):
    # Assume that matrix is a numpy matrix variable
    # Solve for x in the matrix equation Ax = b
    # Independent variables
    matrixRow = matrix.shape[0]
    matrixCol = matrix.shape[1]
    A = matrix[:, :matrixCol-1]
    # Dependent variables
    b = matrix[:, matrixCol-1]
    # A^T
    ATranspose = A.transpose()
    # (A^t * A)
    productAATranspose = np.matmul(ATranspose, A)
    # (A^t * A) ^ -1
    inverseProduct = np.linalg.inv(productAATranspose)
    # (A^t * b)
    productATransposeb = np.matmul(ATranspose, b)
    # x = (A^t * A)^-1 * (A^t * b)
    x = np.matmul(inverseProduct, productATransposeb) # Vector containing the correlation coefficient
    return x

# Take in polynomial order and one-dimensional input data,
# Convert it to multi-linear data,
# Perform regression,
# Return the model parameters and correlation coefficient (computed in linearized space)
def polynomialregression(polynomialOrder, oneDInput, oneDOutput):
    # ------------- CONVERTING FROM POLYNOMIAL TO MULTI-LINEAR ------------------------
    # Convert the one dimensional input data to a matrix to get the row and column
    oneDInput = np.matrix(oneDInput)
    oneDOutput = np.matrix(oneDOutput)

    oneDInputRows = oneDInput.shape[0]
    oneDInputCols = oneDInput.shape[1]
    oneDOutputRows = oneDOutput.shape[0]
    oneDOutputCols = oneDOutput.shape[1]

    # If oneDInput is a row matrix, convert it into a column matrix by transposing
    if(oneDInputCols > 1 and oneDInputRows == 1):
        oneDInput = oneDInput.transpose()
    # If oneDInput is a row matrix, convert it into a column matrix by transposing
    if(oneDOutputCols > 1 and oneDInputRows == 1):
        oneDOutput = oneDOutput.transpose()

    oneDInputRows = oneDInput.shape[0]
    oneDInputCols = oneDInput.shape[1]
    oneDOutputRows = oneDOutput.shape[0]
    oneDOutputCols = oneDOutput.shape[1]

    # Initialize A
    A = np.ones((oneDInputRows, 1)) #For the ^0
    # Initialize B
    b = oneDOutput
    # Modify A
    for i in range (1, polynomialOrder + 1):
        tempA = np.power(oneDInput, i)
        A = np.hstack((A, tempA))

    # ------------- REGRESSION -----------------------
    # Combine A and B to be used for linearregression method
    combinedMatrix = np.hstack((A,b))
    x = linearregression(combinedMatrix)
    return x

# Take in the data,
# Linearize the data,
# Call the linearregression and return the model parameters and correlation coefficient (computed in linearized space)
def exponentialregression():
    return

# Validate the data
a = np.array([0.1, 1.5, 2.5])
x = np.arange(1,5)
y = a[0] + (a[1] * x[:]) + (a[2] * (x[:]**2))
print(y.dtype)
polynomialOrder = 2
aPoly = polynomialregression(polynomialOrder, x, y)
yPoly = aPoly[0] + (aPoly[1] * x[:]) + (a[2] * (x[:]**2))
a = a.flatten()
aPoly = aPoly.flatten()
print("The original coefficients, a, are: ", a)
print("The poly derived coefficients, a, are: ", aPoly)
correlationCoefficient = np.corrcoef(a,aPoly)[1,0] #Or [0,1]
print("The correlation coefficient is: ", correlationCoefficient)
# Apply function to a noisy data set
yNoisy = y + 2 * (np.random.random(x.shape) - 0.5)
aPoly = polynomialregression(polynomialOrder, x, yNoisy)
print("The poly derived coefficients, a, with noise are: ", aPoly)

y = y.reshape(4,1)
yPoly = yPoly.reshape(4,1)
yNoisy = yNoisy.reshape(4,1)

# Plot data obtained my function and theoretical data on x-y axis
plt.figure()
plt.plot(x,y)
plt.plot(x,yPoly)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparing coefficients")
plt.legend(["Theoretical","MyFunction"], loc="best")
plt.show()
# Plot noisy data and theoretical data on x-y axis
plt.figure()
plt.plot(x,y)
plt.plot(x,yNoisy)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Noisy vs Theoretical")
plt.legend(["Theoretical","Noisy"], loc="best")
plt.show()
