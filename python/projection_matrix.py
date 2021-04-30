import numpy as np
import argparse
import glob

import fundamental_matrix as fm


parser = argparse.ArgumentParser(description='Compute Projection Matrix')

parser.add_argument('--input', type=str, help='image input directory', default='../data/model-house/')

args = parser.parse_args()

IMGPATH = args.input


def K():
    return np.array([[-666.265, -1.913, 399.012], [0.0, 672.745, 265.964], [0.0, 0.0, 1.0]])

def W():
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

def Z():
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

def P():
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


def compute_essential_matrix(K, F):
    KT = K.T # KT'FK

    return KT.dot(F).dot(K)


def decompose_essential_matrix(E):
    u,s,vt = np.linalg.svd(E)

    u = -1*u if np.linalg.det(u) < 0 else u
    vt = -1*vt if np.linalg.det(vt) < 0 else vt

    S  = u.dot(Z()).dot(u.T)
    T  = u[:,-1]
    R1 = u.dot(W()).dot(vt)
    R2 = u.dot(W().T).dot(vt)

    return T, R1, R2


def triangulate_X(p1, p2, P1, P2):
    A = np.array([p1.item(0)*P1[2,:] - P1[0,:], p1.item(1)*P1[2,:] - P1[1,:], p2.item(0)*P1[2,:] - P2[0,:], p2.item(1)*P2[2,:] - P2[1,:]])

    U, S, VT = np.linalg.svd(A)

    X = (1/VT[3].item(3)) * VT[3]

    return X


def distance(x1, y1, x2, y2):
    
    d = (x1 - x2)**2 + (y1 - y2)**2 

    return d**0.5


def calculate_error_singular(P1, P2, point):
    p11 = np.array([point.item(0), point.item(1), 1.0])
    p12 = np.array([point.item(2), point.item(3), 1.0])

    X = triangulate_X(p11, p12, P1, P2)

    p21 = P1.dot(X.T)[:-1]
    p22 = P2.dot(X.T)

    p21 = (1/p21.item(2)) * p21
    p22 = (1/p22.item(2)) * p22

    error = distance(p11.item(0), p21.item(0), p11.item(1), p21.item(1))
    error += distance(p12.item(0), p22.item(0), p12.item(1), p22.item(1))

    return error
    

def calculate_error(projection_matrix, points):

    error = 0
    for point in points:
        error += calculate_error_singular(P(), projection_matrix, point)

    return error


def construct_projection_matrix(T,R1,R2,points):

    P = []
    projection_error = 999999999

    projection_matrices = {"P1" : np.column_stack((R1,T)), "P2" : np.column_stack((R1,-T)), "P3" : np.column_stack((R2,T)), "P4" : np.column_stack((R2,-T))}

    for key in projection_matrices.keys():
        error = calculate_error(projection_matrices[key], points)

        if error < projection_error:
            projection_error = error
            P = projection_matrices[key]

    return P


def main():

    images = [fm.read_image(image) for image in sorted(glob.glob(IMGPATH + "house.*.pgm"))]

    sift_keypoints, sift_descriptors = fm.sift_descriptor(images)
    sift_matches = fm.calculate_image_matches(sift_descriptors)

    F, inliers= fm.ransac(fm.find_correspondences(sift_keypoints, sift_matches), 0.95)

    E = compute_essential_matrix(K(), F)

    T, R1, R2 = decompose_essential_matrix(E)

    P = construct_projection_matrix(T, R1, R2, inliers)

    print("Projection Matrix : ", P)

if __name__ == "__main__":
    main()