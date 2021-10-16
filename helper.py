def hello() -> str:
    return "world"

def half(x):
    return (i//2 for i in x)

def cv2pil(x):
    import cv2
    from PIL import Image
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = Image.fromarray(x)
    return x

# def cv2_transfer_size(src, dest):
#     import cv2
#     cv2.resize

def cv2_same_size(a,b):
    import cv2
    maxH = max(a.shape[0], b.shape[0])
    maxW = max(a.shape[1], b.shape[1])
    a = cv2.resize(a, (maxW, maxH))
    b = cv2.resize(b, (maxW, maxH))
    return a,b

def multiply_nn_mnn(g, rgb):
    rgb[:,:,0] = rgb[:,:,0] * g
    rgb[:,:,1] = rgb[:,:,1] * g
    rgb[:,:,2] = rgb[:,:,2] * g

    return rgb