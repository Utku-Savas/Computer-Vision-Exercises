import glob
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Compute Fundamental Matrix')

parser.add_argument('--input', type=str, help='image input directory', default='../data/model-house/')
parser.add_argument('--threshold', type=int, help='error threshold', default=1e-3)
parser.add_argument('--maxiter', type=int, help='maximum number of iterations', default=1000)

args = parser.parse_args()

IMGPATH   = args.input
MAXITER   = args.maxiter
THRESHOLD = args.threshold


def read_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    image_name = image_path.split("/")[-1]
    
    if image is not None:
        print("\t{} successfully read!".format(image_name))
        return image
    
    print("\t{} is invalid!".format(image_name))
    return None


def sift_descriptor(gray):
    sift = cv2.SIFT_create()
    kp_list  = []
    dsc_list = []
    
    for img in gray:
        kp, dsc = sift.detectAndCompute(img, None)
        kp_list.append(kp)
        dsc_list.append(dsc)

    return kp_list, dsc_list


def calculate_image_matches(dsc):

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(dsc[0], dsc[1], k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
    
    return good


def find_correspondences(sift_keypoints, sift_matches):

    correspondence_list = []

    for m in sift_matches:
        x1, y1 = sift_keypoints[0][m.queryIdx].pt
        x2, y2 = sift_keypoints[1][m.trainIdx].pt

        correspondence = [x1, y1, x2, y2]
        
        if correspondence not in correspondence_list:
            correspondence_list.append(correspondence)
    
    correspondence_list = np.array(correspondence_list)

    return correspondence_list


def find_fundamental_matrix(correspondence_list):
    A = []

    for correspondence in correspondence_list:
        x1, y1 = correspondence.item(0), correspondence.item(1)
        x2, y2 = correspondence.item(2), correspondence.item(3)
        
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    A = np.array(A)

    return A

def return_f_vectors(correspondence_list):

    A = find_fundamental_matrix(correspondence_list)

    u, s, vh = np.linalg.svd(A)

    f1 = vh.T[:, -2].reshape(3, 3)
    f2 = vh.T[:, -1].reshape(3, 3)

    return f1, f2


def determinant(f1, f2):
    F = [f1, f2]
    
    return [np.linalg.det(np.array([F[i][:, 0].tolist(), F[k][:, 1].tolist(), F[j][:, 2].tolist()])) for j in range(2) for k in range(2) for i in range(2)]


def find_roots(f1, f2):

    D = determinant(f1, f2)

    c0 = D[3] + D[0] + D[6] + D[5] - D[2] - D[1] - D[7] - D[4]
    c1 = D[1] + D[4] + D[2] + 3*D[7] - 2*D[3] - 2*D[5]  - 2*D[6]
    c2 = D[6] + D[3] + D[5] - 3*D[7]
    c3 = D[7]

    roots = []
    for root in np.roots([c0, c1, c2, c3]):
        if np.isreal(root):
            roots.append(np.real(root))

    return roots


def random_correspondences(correspondences):
    corr1 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr2 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr3 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr4 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr5 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr6 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
    corr7 = correspondences[np.random.randint(len(correspondences), size=1)[0]]

    return [corr1, corr2, corr3, corr4, corr5, corr6, corr7]


def fundamental_error(x1, y1, x2, y2, fundamental):

    p1 = np.array([int(round(x1)), int(round(y1)), 1])
    p2 = np.array([int(round(x2)), int(round(y2)), 1])

    return abs(p2.T.dot(fundamental).dot(p1))


def ransac(correspondences, estimation_thresh):
    inliers = []
    F = []
    len_corr = len(correspondences)


    for i in range(MAXITER):
        seven_correspondences = random_correspondences(correspondences)

        f1, f2 = return_f_vectors(seven_correspondences)
        
        roots = find_roots(f1, f2)

        for a in roots:
            ftmp = a*f1 + (1- a)*f2

            I = []

            for k in range(len_corr):
                corr_temp = correspondences[k]

                x1, y1, x2, y2 = corr_temp.item(0), corr_temp.item(1), corr_temp.item(2), corr_temp.item(3)

                error  = fundamental_error(x1, y1, x2, y2, ftmp)
                

                if error < THRESHOLD:
                    I.append(corr_temp)
            
            if len(I) > len(inliers):
                inliers = I
                F = ftmp
                

            if len(inliers) > estimation_thresh*len_corr:
                break
    
    return F, inliers
        


def main():
    images = [read_image(image) for image in sorted(glob.glob(IMGPATH + "house.*.pgm"))]
    
    sift_keypoints, sift_descriptors = sift_descriptor(images)

    sift_matches = calculate_image_matches(sift_descriptors)

    F, _= ransac(find_correspondences(sift_keypoints, sift_matches), 0.95)

    print("Fundamental Matrix: ", F)

    

if __name__ == "__main__":
    main()