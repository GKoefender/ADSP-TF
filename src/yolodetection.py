import os
import cv2
from Yolo_ADSPTF.detect import run as detection

def draw_bbxs(img, coords, name):
    # Draw the detect object
    pt1 = (coords[0], coords[1])
    pt2 = (coords[0] + coords[2], coords[1] + coords[3])
    cv2.rectangle(img, pt1, pt2, (0,0,255), 2)
    
    ppt1 = (pt1[0],pt1[1]-20)
    ppt2 = (pt1[0]+25,pt1[1])
    cv2.rectangle(img, ppt1, ppt2, (0,0,255), -1)
    
    # Draw the name class of the object
    ppt_name = (pt1[0], pt1[1]-10)
    cv2.putText(img, name, ppt_name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0))

    return img

def yolo(img, final):
    
    # Create temp image dir for yolo inference
    if not os.path.exists('./temp'):
        print('Creating temporary image dir')
        os.makedirs('./temp')    
    
    # Save image for yolo inference    
    cv2.imwrite('./temp/image.jpg', img)

    # Inference of the image with the yolov5 model with pre trained weights
    detections = detection(
        weights='./Yolo_ADSPTF/weights/kitti.pt', source='./temp/image.jpg', 
        project='./temp/', name='detections', save_txt=False, save_conf=True, nosave=True
    )  

    if len(detections) > 0:
        for det in detections:
            det_class = det[1]
            x_n, y_n, w_n, h_n = det[0] # normalized
            
            # Unormalize coordinates
            x = x_n * img.shape[1]
            y = y_n * img.shape[0]
            w = w_n * img.shape[1]
            h = h_n * img.shape[0]

            # Get the top x and y
            x = x - (w/2)
            y = y - (h/2)

            final = draw_bbxs(final, list(map(int, [x,y,w,h])), det_class)

    return final

    

    


    

