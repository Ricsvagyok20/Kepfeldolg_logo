import cv2 as cv

from chamfer_matching.resize import resize_with_aspect_ratio

def chamfer_template(image_path):
    # Load and preprocess the template
    template_image = cv.imread(image_path)
    template_image = resize_with_aspect_ratio(template_image, 512)
    grayscale_template = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
    template_edges = cv.Canny(grayscale_template, 200, 300) # Jó lenne megérteni hogy működik ez, mert most 200 300-al sokkal jobb peugeot template készül

    _, binary_template = cv.threshold(template_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    name = image_path.split('/')[2]

    cv.imwrite('assets/chamfer_templates/' + name + '_chamfer_template.png', binary_template)

    return binary_template


def generate_scaled_templates(template, scales=[0.25, 0.5, 0.75, 1.0]):
    # Generate resized templates based on given scales
    # Érdekes tanulság, hogy mivel különböző képeken különböző méretekben van a logó eredetileg
    # Ezért ha pl nincs 0.25ös akkor ahol nagyobb a logó sokkal jobban megtalálja a logót, de ha van 0.25 akkor talál fals helyeket
    # Báááár most az jutott eszembe, hogy mivan, ha azért lesz kicsi a sum, mert sokkal kevesebb értéket hasonlít össze kisebb template esetén
    # Meg kéne próbálni mondjuk leosztani a scoret a width * height-al
    # Esetleg, hogy még biztosabb legyen a dolog vagy így mostmár több scalet beletenni vagy kicsit nagyobb képet kivágni mint a matchelt terület
    templates = []
    for scale in scales:
        resized_template = resize_with_aspect_ratio(template, 512*scale)
        templates.append(resized_template)
    return templates