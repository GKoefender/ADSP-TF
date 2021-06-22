import cv2
import numpy as np
import math
import argparse
from src.regionOfInterest import regionOfInterest


avgLeft = (0, 0, 0, 0)
avgRight = (0, 0, 0, 0)


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1


def movingAverage(avg, new_sample, N=20):
    if (avg == 0):
        return new_sample
    avg -= avg / N
    avg += new_sample / N
    return avg


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    global avgLeft
    global avgRight

    largestLeftLineSize = 0
    largestRightLineSize = 0
    largestLeftLine = (0, 0, 0, 0)
    largestRightLine = (0, 0, 0, 0)

    if lines is None:
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [
                 255, 255, 255], 12)  # left line
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)), (int(avgx2), int(avgy2)), [
                 255, 255, 255], 12)  # right line
        print('No linesHough')
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2-y1)/(x2-x1))
            if (slope > 0.5):
                if (size > largestRightLineSize):
                    largestRightLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            elif (slope < -0.5):
                if (size > largestLeftLineSize):
                    largestLeftLine = (x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    upLinePoint1 = np.array([0, int(imgHeight - (imgHeight/3))])
    upLinePoint2 = np.array([int(imgWidth), int(imgHeight - (imgHeight/3))])
    downLinePoint1 = np.array([0, int(imgHeight)])
    downLinePoint2 = np.array([int(imgWidth), int(imgHeight)])

    p3 = np.array([largestLeftLine[0], largestLeftLine[1]])
    p4 = np.array([largestLeftLine[2], largestLeftLine[3]])
    upLeftPoint = seg_intersect(upLinePoint1, upLinePoint2, p3, p4)
    downLeftPoint = seg_intersect(downLinePoint1, downLinePoint2, p3, p4)
    if (math.isnan(upLeftPoint[0]) or math.isnan(downLeftPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)),
                 (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)),
                 (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        return
    cv2.line(img, (int(upLeftPoint[0]), int(upLeftPoint[1])), (int(
        downLeftPoint[0]), int(downLeftPoint[1])), [0, 0, 255], 8)

    avgx1, avgy1, avgx2, avgy2 = avgLeft
    avgLeft = (movingAverage(avgx1, upLeftPoint[0]), movingAverage(avgy1, upLeftPoint[1]), movingAverage(
        avgx2, downLeftPoint[0]), movingAverage(avgy2, downLeftPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgLeft
    cv2.line(img, (int(avgx1), int(avgy1)),
             (int(avgx2), int(avgy2)), [255, 255, 255], 12)

    p5 = np.array([largestRightLine[0], largestRightLine[1]])
    p6 = np.array([largestRightLine[2], largestRightLine[3]])
    upRightPoint = seg_intersect(upLinePoint1, upLinePoint2, p5, p6)
    downRightPoint = seg_intersect(downLinePoint1, downLinePoint2, p5, p6)
    if (math.isnan(upRightPoint[0]) or math.isnan(downRightPoint[0])):
        avgx1, avgy1, avgx2, avgy2 = avgLeft
        cv2.line(img, (int(avgx1), int(avgy1)),
                 (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        avgx1, avgy1, avgx2, avgy2 = avgRight
        cv2.line(img, (int(avgx1), int(avgy1)),
                 (int(avgx2), int(avgy2)), [255, 255, 255], 12)
        return
    cv2.line(img, (int(upRightPoint[0]), int(upRightPoint[1])), (int(
        downRightPoint[0]), int(downRightPoint[1])), [0, 0, 255], 8)

    avgx1, avgy1, avgx2, avgy2 = avgRight
    avgRight = (movingAverage(avgx1, upRightPoint[0]), movingAverage(avgy1, upRightPoint[1]), movingAverage(
        avgx2, downRightPoint[0]), movingAverage(avgy2, downRightPoint[1]))
    avgx1, avgy1, avgx2, avgy2 = avgRight
    cv2.line(img, (int(avgx1), int(avgy1)),
             (int(avgx2), int(avgy2)), [255, 255, 255], 12)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def main(videoDir, debug=False):    
    video = cv2.VideoCapture(videoDir)
    while True:
        ret, orig_frame = video.read()
        if not ret:
            video = cv2.VideoCapture(videoDir)
            continue

        # Gaussian blur
        blurImg_5 = cv2.GaussianBlur(orig_frame, (5, 5), 0)
        blurImg_11 = cv2.GaussianBlur(orig_frame, (11, 11), 0)
        blurImg_114 = cv2.GaussianBlur(orig_frame, (11, 11), cv2.BORDER_DEFAULT)

        # Detecção de bordas de Canny
        edgesImage_5 = cv2.Canny(blurImg_5, 30, 40)
        edgesImage_11 = cv2.Canny(blurImg_11, 30, 40)
        edgesImage_114 = cv2.Canny(blurImg_114, 40, 50)

        image = edgesImage_11

        regionInterestImage = regionOfInterest(image, "trapezio")

        # lines = cv2.HoughLinesP(regionInterestImage, 1, np.pi/180, 40, np.array([]), 30, 200)
        # line_img = np.zeros((*regionInterestImage.shape, 3), dtype=np.uint8)

        #lineMarkedImage = hough_lines(regionInterestImage, 1, np.pi/180, 40, 30, 200)
        lineMarkedImage = hough_lines(
            regionInterestImage, 1, np.pi/180, 50, 50, 10)

        # correção de fluxo
        # image = cv2.line(image, start_point, end_point, color, thickness)
        height = lineMarkedImage.shape[0]
        width = lineMarkedImage.shape[1]

        correctionImage = cv2.line(lineMarkedImage, (int(
            width/2), height), (int(width/2), int(0.9*height)), (100, 100, 100), 5)

        finalImage = weighted_img(correctionImage, orig_frame)

        if debug:
            cv2.imshow("orig_frame", orig_frame)
            #cv2.imshow("edgesImage_11", edgesImage_11)
            #cv2.imshow("edgesImage_114", edgesImage_114)
            cv2.imshow("correctionImage", correctionImage)
        
        cv2.imshow("finalImage", finalImage)
        #cv2.imshow("orig_frame", orig_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--debug', action='store_true', help='Show all the layers of processing image')
    
    args = parser.parse_args()
    
    main(args.path, args.debug)