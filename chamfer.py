import cv2 as cv

def chamfer_template():
    template_image = cv.imread('assets/logos/honda/honda_logo_main_for_chamfer.png')
    grayscale_template = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
    template_edges = cv.Canny(grayscale_template, 300, 550) #Itt majd ezeket a paramétereket lehet állítani

    # 300 550 elég jó

    _, binary_template = cv.threshold(template_edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Előfeldolgozás-hoz még hozzá lehet adni a kontraszt növelést vagy nemtudom mivel lehetne javítani pl. a 9-es honda logot
    # Majd meglátjuk, hogy kell-e extra zajszűrés

    cv.imshow("Eredeti kép:", template_image)
    # cv.imshow("Élek kép:", template_edges)
    cv.imshow("Binary kép:", binary_template)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # cv.imwrite('assets/chamfer_templates/honda_chamfer_template.png', template_edges)


# print("\nTesztelés...")
#
# # Jellemző vektorok előállítása
# print("Tesztminták feldolgozása:")
# testData = []
# print("Villák...")
# forks = process_data("FORK/*.png", 'test')
# testData.append(forks)
# labels_1 = np.full(len(forks), 1)

chamfer_template()