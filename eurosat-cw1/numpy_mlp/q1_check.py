"""
Numerical verification script for NB6057CEM CW2 Theory Report Question 1.
Computes forward pass, exact analytical gradients via chain rule, simultaneous
gradient descent parameter updates, and post-update forward pass evaluation.
"""

import numpy as np

def run_q1_verification():
    # 1. Inputs and target
    x = np.array([1.0, 2.0])
    y = 1.0

    # 2. Initial parameters
    W = np.array([[0.2, -0.1],
                  [0.1,  0.3]])
    b = np.array([0.1, -0.2])
    v = np.array([0.4, -0.5])
    c = 0.2
    eta = 0.1

    print("=== Q1(b) FORWARD PASS ===")
    z = W @ x + b
    h = np.maximum(0, z)  # ReLU activation
    y_hat = np.dot(v, h) + c
    error = y_hat - y
    loss = 0.5 * (error ** 2)

    print(f"Pre-activations z : [{z[0]:.4f}, {z[1]:.4f}]")
    print(f"Activations h     : [{h[0]:.4f}, {h[1]:.4f}]")
    print(f"Prediction y_hat  : {y_hat:.4f}")
    print(f"Prediction error e: {error:.4f}")
    print(f"Scalar loss L     : {loss:.6f}")

    print("\n=== Q1(c) BACKWARD PASS (CHAIN RULE) ===")
    dL_dyhat = error
    dL_dc = dL_dyhat
    dL_dv = dL_dyhat * h

    # ReLU derivative indicator (z > 0)
    dh_dz = (z > 0).astype(float)
    delta_h = (dL_dyhat * v) * dh_dz

    dL_db = delta_h
    dL_dW = np.outer(delta_h, x)

    print(f"dL/dc  = {dL_dc:.4f}")
    print(f"dL/dv1 = {dL_dv[0]:.4f}, dL/dv2 = {dL_dv[1]:.4f}")
    print(f"delta_h1 = {delta_h[0]:.4f}, delta_h2 = {delta_h[1]:.4f}")
    print(f"dL/db1 = {dL_db[0]:.4f}, dL/db2 = {dL_db[1]:.4f}")
    print(f"dL/dW11 = {dL_dW[0,0]:.4f}, dL/dW12 = {dL_dW[0,1]:.4f}")
    print(f"dL/dW21 = {dL_dW[1,0]:.4f}, dL/dW22 = {dL_dW[1,1]:.4f}")

    print("\n=== SIMULTANEOUS GRADIENT DESCENT UPDATE (eta = 0.1) ===")
    c_new = c - eta * dL_dc
    v_new = v - eta * dL_dv
    b_new = b - eta * dL_db
    W_new = W - eta * dL_dW

    print(f"c_new  = {c_new:.4f}")
    print(f"v1_new = {v_new[0]:.4f}, v2_new = {v_new[1]:.4f}")
    print(f"b1_new = {b_new[0]:.4f}, b2_new = {b_new[1]:.4f}")
    print(f"W_new  :\n{W_new}")

    print("\n=== POST-UPDATE RECOMPUTATION ===")
    z_new = W_new @ x + b_new
    h_new = np.maximum(0, z_new)
    y_hat_new = np.dot(v_new, h_new) + c_new
    error_new = y_hat_new - y
    loss_new = 0.5 * (error_new ** 2)

    print(f"New pre-activations z : [{z_new[0]:.4f}, {z_new[1]:.4f}]")
    print(f"New activations h     : [{h_new[0]:.4f}, {h_new[1]:.4f}]")
    print(f"New prediction y_hat  : {y_hat_new:.6f}")
    print(f"New prediction error e: {error_new:.6f}")
    print(f"New scalar loss L     : {loss_new:.6f}")
    print(f"Loss reduction delta L: {loss - loss_new:.6f}")

if __name__ == "__main__":
    run_q1_verification()
