from PIL import ImageChops, ImageFilter, Image
import numpy as np
from helper import cv2_same_size, multiply_nn_mnn

def reduce0(im, kernel):
    im = im.filter(kernel) # Blur the image
    im = im.resize((i//2 for i in im.size)) # Downsample the image
    return im

def pyramid(img, scale=5, kernel = ImageFilter.GaussianBlur(15)) -> list:
    # When building gaussian pyramids, the original image is part of the pyramid
    imgs = [img] #img.filter(kernel)
    prev_img = img
    for i in range(scale-1):
        prev_img = reduce0(prev_img, kernel)
        imgs.append(prev_img)
    return imgs

def expand(im):
    return im.resize((i*2 for i in im.size)) # upsame the image by a factor of 2

def laplacian(img, scale=5, kernel = ImageFilter.GaussianBlur(15)) -> list:
    # An improvement could be to accept the pyramid from `pyramid` fcn
    # this would save recomputing the `reduce0` ops repetitively

    result = []
    gn = img
    for s in range(scale):
        gn1 = reduce0(gn, kernel)
        # ln = ImageChops.difference(expand(gn1), gn)
        ln = ImageChops.subtract(gn, expand(gn1))
        result.append(ln)
        gn = gn1

    return result

def laplacian0(gp, scale=5) -> list:
    assert(scale, len(gp), "The scale must equal the pyramid size")
    lp = [gp[-1]] # the tip of the pyramid is always taken from the tip of the gaussian pyramid
    for i in reversed(range(scale-1)):
        print(f"Scale {i}")
        gExp = expand(gp[i+1])
        li = ImageChops.subtract(gp[i], gExp)
        lp.insert(0, li)
    return lp

"""
`result` is an image pyramid. An image pyramid created using the alpha matting equation where
    * the alpha matte is gaussian pyramid of the mask
    * The other 2 pyramids a created from the laplacian pyramids of image A & B to be blended.
"""
def reconstruct_laplacian(result):
    scale = len(result)
    up = result[-1]
    for i in range(scale-1,0,-1):
        up = up.resize(result[i-1].size, Image.ANTIALIAS)
        up = ImageChops.add(result[i-1], up)
    return up

def cv_multiresolution_blend(gm, la, lb) -> list:
    gm = [x // 255 for x in gm]
    blended = []
    for i in range(len(gm)):
        gmi , lbi = cv2_same_size(gm[i], lb[i])
        bi = multiply_nn_mnn(gmi, lbi) + multiply_nn_mnn((1-gmi), la[i])
        bi = bi.astype(np.uint8)
        blended.append(bi)
    return blended

def cv_reconstruct_laplacian(blended_pyramid):
    import cv2
    scale = len(blended_pyramid)
    up = blended_pyramid[-1] # start with the tip, this is would the smallest scale image, while the rest of the pyramid would contain blended laplacians
    for i in range(scale-1, 0, -1):
        next = blended_pyramid[i-1].copy()
        up = cv2.pyrUp(up)
        print(f"{next.shape} - {up.shape}")
        up, next = cv2_same_size(up, next) #sometimes the width/height can be off by a few pixels due to `cv2.pyrUp`
        up = cv2.add(next, up)

    return up

def smoothing_kernel() -> ImageFilter.Kernel:
    kernel_size = (3,3)
    # kernels are defined row-wise
    kernel = [\
        1, 2, 1,\
        2, 4, 2,\
        1, 2, 1]
    return ImageFilter.Kernel(kernel_size, kernel)

def cv_pyramid(A, scale):
    import cv2
    gp = [A]
    for i in range(1, scale):
        A = cv2.pyrDown(A)
        gp.append(A)
    return gp

def cv_laplacian(gp, scale) -> list:
    import cv2
    lp = [gp[-1].copy()]
    for i in reversed(range(scale-1)):
        gExp = cv2.pyrUp(gp[i+1].copy())
        gpi = gp[i].copy()
        gpi, gExp = cv2_same_size(gpi, gExp)
        li = cv2.subtract(gpi, gExp)
        lp.insert(0, li)
    return lp