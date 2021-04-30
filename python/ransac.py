import glob
import cv2
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Feature Detection and Matching')

parser.add_argument('--input', type=str, help='image directory', default='../data/goldengate/')
parser.add_argument('--distance', type=int, help='ransac distance error threshold', default=50)
parser.add_argument('--maxiter', type=int, help='maximum number of iterations', default=1000)

args = parser.parse_args()

DATA = args.input
DIST_THRESHOLD = args.distance
MAXITER = args.maxiter


def read_image(image_path):
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1]
    
    if image is not None:
        print("{} successfully read!".format(image_name))
        return image
    
    print("{} is invalid!".format(image_name))
    return None


def distance(x1, y1, x2, y2):
    d = (x1 - x2)**2 + (y1 - y2)**2 
    return d**0.5


def find_estimation_points(x1, y1, homography):
    p = np.array([int(round(x1)), int(round(y1)), 1])
    p = np.dot(homography, p.T)
    p = (1/p.item(2)) * p
    return int(round(p.item(0))), int(round(p.item(1)))


def homography_error(x1, y1, x2, y2, homography):
    
    e_x, e_y = find_estimation_points(x1, y1, homography)

    return distance(x2 ,y2, e_x, e_y)


def is_similar(correspondence_list):

    for index, cor1 in enumerate(correspondence_list):
        x11, y11, x12, y12 = cor1.item(0), cor1.item(1), cor1.item(2), cor1.item(3)

        for cor2 in correspondence_list[index+1:]:
            x21, y21, x22, y22 = cor2.item(0), cor2.item(1), cor2.item(2), cor2.item(3)

            if [x11, y11] == [x21, y21] or [x12, y12] == [x22, y22]:
                return True
    
    return False


def return_kp_and_desc_list(gray_image):

    sift = cv2.SIFT_create()

    return sift.detectAndCompute(gray_image, None)


def bf_matcher(desc1, desc2):

    bf = cv2.BFMatcher()
    
    matches = bf.knnMatch(desc1, desc2, k=2)
    print("Before ratio test :: Number of matches : {}".format(len(matches)))
    
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    print("After ratio test :: Number of matches : {}".format(len(good)))
    
    return good


def flann_matcher(desc1, desc2):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    matches = flann.knnMatch(desc1, desc2, k=2)
    print("Before ratio test :: Number of matches : {}".format(len(matches)))
    
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            
    print("After ratio test :: Number of matches : {}".format(len(good)))
    
    return good


def find_correspondences(kp1, kp2, matches):

    correspondences = []
    
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        correspondences.append([x1, y1, x2, y2])
        
    print("Number of correspondences : {}".format(len(correspondences)))
    return np.array(correspondences)


def find_homography(correspondence_list):

    A = []

    for correspondence in correspondence_list:
        x1, y1 = correspondence.item(0), correspondence.item(1)
        x2, y2 = correspondence.item(2), correspondence.item(3)
        
        A.append([-1*x1, -1*y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -1*x1, -1*y1, -1, x1*y2, y1*y2, y2])
    
    A = np.array(A)

    U, S, VT = np.linalg.svd(A)

    H = (1/VT[8].item(8)) * VT[8]

    return np.reshape(H, (3, 3))


def ransac(correspondences, estimation_thresh):

    inliers = []
    homography = []

    len_corr = len(correspondences)


    for i in range(MAXITER):
        print("Progress {:2.1%}".format(i / MAXITER), end="\r")

        corr1 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
        corr2 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
        corr3 = correspondences[np.random.randint(len(correspondences), size=1)[0]]
        corr4 = correspondences[np.random.randint(len(correspondences), size=1)[0]]

        if is_similar([corr1, corr2, corr3, corr4]):
            continue
        else:
            H = find_homography([corr1, corr2, corr3, corr4])
        
            HI = np.linalg.inv(H)
            HI = (1/HI.item(8)) * HI

            I = []

            for k in range(len_corr):
                corr_temp = correspondences[k]

                x1, y1, x2, y2 = corr_temp.item(0), corr_temp.item(1), corr_temp.item(2), corr_temp.item(3)

                error  = homography_error(x1, y1, x2, y2, H)
                error += homography_error(x2, y2, x1, y1, HI)

                if error < DIST_THRESHOLD:
                    I.append(corr_temp)
            
            if len(I) > len(inliers):
                inliers = I
                homography = H

            if len(inliers) > estimation_thresh*len_corr:
                break
    
    print(" "*30, end="\r")
    return homography


def draw_keypoints(rgb_image, gray_image, keypoints):
    
    img = cv2.drawKeypoints(gray_image, keypoints, rgb_image)

    cv2.imshow("Sift Keypoints", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_matches(gimg1, kp1, gimg2, kp2, matches):
    
    img = cv2.drawMatches(gimg1, kp1, gimg2, kp2, matches, gimg2, flags=2, matchColor = (0,255,255))
    
    cv2.imshow('Image Matches', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    rgb_images= [read_image(image) for image in sorted(glob.glob(DATA + "goldengate*.png"))[:2]]
    gimg1 = cv2.cvtColor(rgb_images[0], cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(rgb_images[1], cv2.COLOR_BGR2GRAY)
    
    kp1, desc1 = return_kp_and_desc_list(gimg1)
    kp2, desc2 = return_kp_and_desc_list(gimg2)
    
    matches = flann_matcher(desc1, desc2)
    
    correspondences = find_correspondences(kp1, kp2, matches)
    
    homography = ransac(correspondences, 0.95)
    
    print("Homography Matrix: ", homography)
    
    draw_keypoints(rgb_images[0], gimg1, kp1)
    
    draw_matches(gimg1, kp1, gimg2, kp2, matches)


if __name__ == "__main__":
    main()
