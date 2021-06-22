import cv2
import numpy as np


def getVertices(format, height, width):
    if(format == "trapezio"):
        return np.array([[
            [3*width/4, 3*height/4.7],
            [width/4, 3*height/4.7],
            (0, height),
            (width, height)
        ]], dtype=np.int32)

    if(format == "trapezio2"):
        return np.array([[
            [3*width/4, 3*height/5],
            [width/4, 3*height/5],
            [40, height],
            [width - 40, height]
        ]], dtype=np.int32)

    if(format == "triangulo"):
        return np.array([[
            [0, height],
            [width/2, height/2],
            [width, height]
        ]], dtype=np.int32)

    if(format == "retangulo"):
        return np.array([[
            [width, height/2],
            [0, height/2],
            [0, height],
            [width, height]
        ]], dtype=np.int32)

    print('Nenhum formato de regiao encontrado')
    return None


# format: trapezio, trapezio2, triangulo
def regionOfInterest(img, format):
    height = img.shape[0]
    width = img.shape[1]
    vertices = getVertices(format, height, width)
    mask = np.zeros_like(img)

    '''
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    '''

    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    cv2.fillPoly(mask, vertices, 255)

    # retorna a imagen onde os pixels nao sao zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
