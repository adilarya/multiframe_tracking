from PIL import Image
import numpy as np
from cv2 import resize
import matplotlib.pyplot as plt

from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

import random # used for RANdom in RANSAC

def find_match(img1, img2):
    x1, x2 = None, None
    dis_thr = 0.7 # very standard threshold

    # converting to a list to hold coordinates
    x1 = []
    x2 = []

    # getting the keypoints using SIFT demo given
    sift = SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None) # for img1
    kp2, des2 = sift.detectAndCompute(img2, None) # for img2

    # type (cv2.KeyPoint) in kp1 and kp2
    # R128 vectors in des1 and des2
    
    # neighbors _ for _
    n2f1 = NearestNeighbors(n_neighbors=2).fit(des2)
    n1f2 = NearestNeighbors(n_neighbors=2).fit(des1)

    # distances from img1 pt to knn of img2
    dist1to2, idx1to2 = n2f1.kneighbors(X=des1, n_neighbors=2)
    dist2to1, idx2to1 = n1f2.kneighbors(X=des2, n_neighbors=2) # vice versa

    # filtering for dist threshold
    idx1to2f = []
    idx2to1f = []

    for i in range(len(dist1to2)) :
        curr = dist1to2[i]

        if curr[0] / curr[1] < dis_thr :
            idx1to2f.append([i, int(idx1to2[i][0])])

    for i in range(len(dist2to1)) :
        curr = dist2to1[i]

        if curr[0] / curr[1] < dis_thr :
            idx2to1f.append([i, int(idx2to1[i][0])])

    # bidirectional check
    for i in range(len(idx1to2f)) :
        curr = idx1to2f[i]
        swapped = [curr[1], curr[0]]

        if swapped in idx2to1f :
            idx1 = curr[0]
            idx2 = curr[1]

            x1.append(kp1[idx1].pt)
            x2.append(kp2[idx2].pt)

    x1 = np.array(x1)
    x2 = np.array(x2)

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = None

    assert x1.shape[0] >= 3, 'x1 and x2 need to have at least 3 correspondences!' # affine transform requires 3

    s = 3 # randomly choose s samples (6 for affine) | this is # of correspondences
    max_inliers = 0 # initializing at 0

    for i in range(ransac_iter) :
        At = None
        pairs = []
        idxs = []

        while len(pairs) != s : # adding pair if not in pairs (trying to avoid duplicates)
            ridx = random.randint(0, x1.shape[0] - 1)

            if ridx not in idxs :
                idxs.append(ridx)
                pairs.append([x1[ridx], x2[ridx]])

        u2 = [] # final values
        for j in range(len(pairs)) :
            u2.append([pairs[j][1][0]])
            u2.append([pairs[j][1][1]])

        u2 = np.array(u2) # this is getting the column vector for the actual transform.
        # currently we are attempting to get the a11, a12, a13... values to construct the At matrix.

        M = [] # the 6x6 matrix
        for j in range(len(pairs)):
            base = [pairs[j][0][0], pairs[j][0][1], 1]
            empty = [0, 0, 0]

            M.append(base + empty)
            M.append(empty + base)
        
        M = np.array(M)
        
        # now that we have the two general matrices, we can calculate the a11, a12, ...
        # M * b = u2 (we just use the np formulas to solve)
        
        # required for degenerate
        if np.linalg.matrix_rank(M) < 6: 
            break

        b = np.linalg.solve(M, u2)
        
        # now we reconstruct the At matrix
        At = b.reshape(2, 3) # contains the a values (that were missing)
        At = np.append(At, np.array([0,0,1])) # adding last row (currently a 2x3 matrix)
        At = At.reshape(3, 3) # reshaping, because it previous function flattens to (9,)

        # now that we have At (affine transform matrix at curr time), we can test for inliers
        inliers = 0

        for i in range(len(x1)) :
            # getting the 'x' column matrix from the Ax = b equation
            x = [[x1[i][0]], [x1[i][1]], [1]]
            x = np.array(x)
            
            # now getting the predicted 'b_' column matrix from same equation
            b_ = At @ x

            # actual from the x2
            b = [[x2[i][0]], [x2[i][1]], [1]]
            b = np.array(b)

            diff = b - b_

            # error
            diff = diff.reshape(3)
            e = np.linalg.norm(diff)

            if e < ransac_thr :
                inliers += 1

        if inliers > max_inliers :
            max_inliers = inliers 
            A = At
        
    return A

def warp_image(img, A, output_size):
    img_warped = None

    H, W = img.shape # getting for the points attribute for interpn
    vdim = np.arange(H) # row indices [0 1 2 ... H]
    udim = np.arange(W) # col indices [0 1 2 ... W]

    # given
    h = output_size[0]
    w = output_size[1]
    img_warped = np.zeros((h, w))

    # in attempts to optimize, we will matrix multiply all values
    x2 = [[], [], []]

    # this function is to generate the full stack of x2 vectors to transform
    # this way we can matrix multiply all in one shot
    for v2 in range(h) :
        for u2 in range(w) :
            # adding the vector piece by piece
            x2[0].append(u2)
            x2[1].append(v2)
            x2[2].append(1)
    
    x2 = np.array(x2)

    x1 = A @ x2 # doing matrix multiplication in one shot

    x1 = np.delete(x1, 2, axis=0) # getting rid of the 1s (3rd row - redundant)
    # currently, the order of these is a row of [u1 u2 u3 .... un] and [v1 v2 v3 ... vn]
    # the rows need to be swapped, since when we transpose it will otherwise be (u1, v1)
    # and for correct row/col association, it must be (v1, u1). Hence we swap the rows. 
    x1[[0, 1]] = x1[[1, 0]] # VERY IMPORTANT
    x1 = x1.T # now we have a list of all the points in (v1, u1) order

    # getting values 
    vals = interpolate.interpn(points=(vdim, udim),
                                     values=img,
                                     xi=x1,
                                     method='linear',
                                     bounds_error=False,
                                     fill_value=0.0,
                                    )

    vals = np.reshape(vals, (h, w)) # reshaping 1d array to img warped supposed size
    img_warped = vals

    return img_warped

def align_image(template, target, A):
    A_refined = None
    errors = None
    template = template / 255.0 # normalizing 
    target = target / 255.0
    
    h, w = template.shape

    # Initializing p = p0 from input A
    p = np.delete(A, 2, axis=0) 
    p = p.flatten() # making 1D
    p[0] -= 1 # getting p1 = a11 - 1 (since p1 + 1 = a11)
    p[4] -= 1 # same as above.
    
    # Computing deltaI_tpl in deltaI_u and deltaI_v
    filter_u = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float64)
    filter_v = np.array([[1, 2, 1],
                         [0, 0, -0],
                         [-1, -2, -1]], dtype=np.float64)
    
    delta_Iu = filter2D(template, -1, filter_u) # computes correlation, not convolution
    delta_Iv = filter2D(template, -1, filter_v)

    # Computing Jacobian J 
    u, v = np.meshgrid(np.arange(w), np.arange(h)) # creating to setup 2 x 6 Jacobian matrix
    J = np.zeros((h, w, 2, 6), dtype=np.float64)
    J[:, :, 0, 0] = u # dW/dp1
    J[:, :, 0, 1] = v # dW/dp2
    J[:, :, 0, 2] = 1 # dW/dp3
    J[:, :, 1, 3] = u # dW/dp4
    J[:, :, 1, 4] = v # dW/dp5
    J[:, :, 1, 5] = 1 # dW/dp6

    # Computing Steepest Descent Images (SDI)
    SDI = np.zeros((h, w, 6), dtype=np.float64)
    for i in range(6) :
        SDI[:, :, i] = delta_Iu * J[:, :, 0, i] + delta_Iv * J[:, :, 1, i] 
        # doing the (1 x 2) . (2 x 6) multiplication for each pixel

    SDI_reshaped = SDI.reshape(-1, 6) 
    SDI_sample = SDI_reshaped[:, :].astype(np.float64)

    # Computing Hessian H
    H = SDI_reshaped.T @ SDI_reshaped # (6, 6)
    
    # Variables for Iteration
    errors = []
    max_iters = 250 # hyperparameter
    e = 1e-4 # hyperparameter

    while True :
        I_warped = warp_image(target, A, output_size=(h, w)).astype(np.float64) # Warping target to template domain
        I_warped = I_warped.reshape(-1) # flattening for F
        I_error = (template.reshape(-1) - I_warped).astype(np.float64) # Computing error image
        F = SDI_sample.T @ I_error # Computing F 
        
        delta_p = np.linalg.solve(H, F) # Computing delta_p
        # assuming that H is invertible here

        # Updating W(x; p) <- W(x; p) o W^-1(x; delta_p) 
        dp = delta_p.flatten()
        delta_A = np.array([[dp[0] + 1.0, dp[1], dp[2]],
                            [dp[3], dp[4] + 1.0, dp[5]],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        
        try: # computing inverse with safeguard
            delta_A_inv = np.linalg.inv(delta_A) 
        except np.linalg.LinAlgError:
            delta_A_inv = np.linalg.pinv(delta_A) # pseudo-inverse as fallback
        
        A = A @ delta_A_inv # updating A

        rt = np.linalg.norm(I_error) # frobenius norm
        SSE = np.sum(rt ** 2)
        RMSE = np.sqrt(np.mean(SSE ** 2))

        errors.append(RMSE) # scale normalized
        if np.linalg.norm(delta_p) < e or len(errors) >= max_iters:
            break

    A_refined = A
    errors = np.array(errors)

    return A_refined, errors

def track_multi_frames(template, img_list):
    A_list = None
    errors_list = None

    ransac_iter = 100
    ransac_thr = 10.0

    A_list = []
    errors_list = []

    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    for i in range(len(img_list)) :
        A_refined, errors = align_image(template, img_list[i], A)
        template = warp_image(img_list[i], A_refined, template.shape) # updating template
        A_list.append(A_refined)
        errors_list.append(errors)
        A = A_refined # updating A for next frame
        print(f'Frame {i+1}/{len(img_list)} processed.')

    A_list = np.array(A_list)
    errors_list = np.array(errors_list)

    return A_list, errors_list

# ----- Visualization Functions -----

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    import cv2
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

def visualize_track_multi_frames(template, img_list, A_list, errors_list=None):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

    if errors_list is not None:
        for i, errors in enumerate(errors_list):
            plt.plot(errors * 255)
            plt.title(f'Frame {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.show()

# ----- Visualization Functions -----

if __name__=='__main__':

    template = Image.open('template.jpg')
    template = np.array(template.convert('L'))
    
    target_list = []
    for i in range(4):
        target = Image.open(f'target{i+1}.jpg')
        target = np.array(target.convert('L'))
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 10 # hyperparameter
    ransac_iter = 100 # hyperparameter
    # ----------

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list, errors_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list, errors_list)