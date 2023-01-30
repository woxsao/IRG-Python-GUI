import numpy as np

# Test case 1
A_g = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
x = np.array([[1, 2, 3], [4, 5, 6]])
b_g = np.array([[1, 2], [3, 4]])
beta_k_x = np.array([[1, 2, 3], [4, 5, 6]])
k = 0
i = 0
f_g = np.zeros((2,3))
print(A_g[k,:,:])
print(np.transpose(x[:,i]).shape)
f_g[:,k] = np.multiply(beta_k_x[k,i], (np.matmul(A_g[:,:,k], x[:,i]) + b_g[:,k]))
print(f_g) # Expected output: [[8. 0. 0.]
            #                    [22. 0. 0.]]

