import cv2 as cv


def dobozolas(image, best_chamfer_location, best_template_size):
    top_left = best_chamfer_location
    h, w = best_template_size
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    cv.imshow("Best Match", image)
    cv.waitKey(0)
    cv.destroyAllWindows()