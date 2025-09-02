import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.linalg import khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
    # Map training data to feature space
    X_mapped = my_map(X_train)
    # Use LogisticRegression with balanced class weights
    clf = LogisticRegression(C=2.0, tol=1e-6, max_iter=3000, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
    clf.fit(X_mapped, y_train)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    return w, b

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    # Convert challenges from {0,1} to {-1,1}
    X_bin = np.flip(2 * X - 1, axis=1)  # MSB-first
    N, D = X.shape
    # Quadratic feature map: [1, x_0, ..., x_7, x_0*x_1, ..., x_6*x_7, triplets, quadruplets]
    phi = np.hstack([np.ones((N, 1)), X_bin])  # Shape: [N, 9] (bias + 8 bits)
    pairwise = []
    for i in range(D):
        for j in range(i + 1, D):
            pairwise.append(X_bin[:, i] * X_bin[:, j])
    pairwise = np.stack(pairwise, axis=1) if pairwise else np.empty((N, 0))
    # Add triplet terms: x_i*x_{i+1}*x_{i+2}, x_i*x_{i+2}*x_{i+4}, x_i*x_{i+1}*x_{i+3}
    triplets = []
    for i in range(D - 2):  # x_i*x_{i+1}*x_{i+2}
        triplets.append(X_bin[:, i] * X_bin[:, i + 1] * X_bin[:, i + 2])
    for i in range(D - 4):  # x_i*x_{i+2}*x_{i+4}
        triplets.append(X_bin[:, i] * X_bin[:, i + 2] * X_bin[:, i + 4])
    for i in range(D - 3):  # x_i*x_{i+1}*x_{i+3}
        triplets.append(X_bin[:, i] * X_bin[:, i + 1] * X_bin[:, i + 3])
    # Add quadruplet terms: x_i*x_{i+1}*x_{i+2}*x_{i+3} (reduced)
    quadruplets = []
    for i in range(3):  # x_i*x_{i+1}*x_{i+2}*x_{i+3} for i=0,1,2
        quadruplets.append(X_bin[:, i] * X_bin[:, i + 1] * X_bin[:, i + 2] * X_bin[:, i + 3])
    higher_order = np.stack(triplets + quadruplets, axis=1) if (triplets or quadruplets) else np.empty((N, 0))
    X_mapped = np.hstack([phi, pairwise, higher_order])  # Shape: [N, 1 + 8 + 28 + 6 + 4 + 5 + 3 = 55]
    return X_mapped

#################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################
   # Split model into weights and bias
    w = w[:-1]
    b = w[-1]

    # Initialize data structures
    rows, cols = 65, 256
    A_matrix = np.zeros((rows, cols))
    y_vector = np.zeros(rows)

    # Configure first row
    A_matrix[0, :4] = [0.5, -0.5, 0.5, -0.5]
    y_vector[0] = w[0]

    # Populate rows 1 to 63
    for row in range(1, rows-1):
        base_idx = row * 4
        prev_idx = (row - 1) * 4

        # Set current row coefficients
        A_matrix[row, base_idx:base_idx+4] = [0.5, -0.5, 0.5, -0.5]
        
        # Update previous row coefficients
        A_matrix[row, prev_idx:prev_idx+4] += [0.5, -0.5, -0.5, 0.5]
        
        y_vector[row] = w[row]

    # Configure last row
    final_base = 63 * 4
    A_matrix[-1, final_base:final_base+4] = [0.5, -0.5, -0.5, 0.5]
    y_vector[-1] = b

    # Train regression model
    model_reg = LinearRegression(positive=True)
    model_reg.fit(A_matrix, y_vector)
    coeffs = np.clip(model_reg.coef_, a_min=0, a_max=None)

    # Slice coefficients into groups
    p_vals = coeffs[::4]
    q_vals = coeffs[1::4]
    r_vals = coeffs[2::4]
    s_vals = coeffs[3::4]

    return p_vals, q_vals, r_vals, s_vals