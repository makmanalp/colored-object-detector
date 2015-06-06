import numpy as np
import cv2

cimport numpy as np
cimport cython

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def num_pixels_in_contour(np.ndarray[DTYPE_t, ndim=2] image, np.ndarray[int, ndim=3] contour):

    cdef np.ndarray[DTYPE_t, ndim=2] mask
    cdef np.ndarray[DTYPE_t, ndim=2] masked_image

    # Grab pixels inside contour
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=DTYPE)
    cv2.fillPoly(mask, [contour], (255, 255, 255))
    masked_image = image * mask

    # Count white pixels
    return cv2.countNonZero(masked_image)
