import cv2 as cv

def chamfer_template(image_path):
    # Load and preprocess the template
    template_image = cv.imread(image_path)
    grayscale_template = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
    template_edges = cv.Canny(grayscale_template, 300, 550)

    _, binary_template = cv.threshold(template_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary_template


def generate_scaled_templates(template, scales=[0.25, 0.5, 1.0, 1.5, 2.0]):
    # Generate resized templates based on given scales
    templates = []
    for scale in scales:
        resized_template = cv.resize(template, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        templates.append(resized_template)
    return templates
