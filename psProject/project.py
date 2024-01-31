"""
Universitatea din Bucuresti
Facultatea de Matematica si Informatica

CTI - Anul 4 - grupa 461

Echipa LOR :
- Codreanu Radu-Stefan
- Albei Liviu-Andrei
- Cojocaru Andrei-Laurentiu

Mod de utilizare:
- este nevoie de cel putin o imagine (ATENTIE!!! extensia trebuie sa fie .jpg) in cadrul folderului in care se afla "project.py", peste care vom putea aplica doar un watermark text
- rulam aplicatia si introducem de la tastatura "1"
- dupa introducere, asteptam un timp scurt, se calculeaza entropia imaginii si, implicit, se afiseaza (observam unde este noise/zgomot mai mare se intensifica culoarea)
- apoi, ne este cerut sa introducem textul pe care il dorim sa fie afisat peste imaginea originala, opacitatea (intre 0 si 1) pentru subpunctul legat de "vizibilitate vs invizibilitate"
- un unghi de rotatie, daca dorim sa fie rotit textul, o pozitionare predefinita (sus, jos, stanga, dreapta, centru // top, bottom, left, right, center)
- pozitionare ce urmeaza a fi modificata, daca utilizatorul doreste, dupa placere, cu o margine iar in final, se introduce culoarea dorita a textului
- imaginea rezultata este salvata in acelasi folder cu "project.py" si cu imaginea originala, sub numele de "result_image.jpg"

- pentru cel de-al doilea caz, cel cu input "2", este nevoie de doua imagini, una originala si una ce va reprezenta watermark-ul ce urmeaza sa fie aplicat peste imaginea originala (utilizant metoda LSB)
- imaginile, ca la prima optiune, trebuie sa aiba extensia .jpg
- se va afisa calculul entropiei si imaginea acesteia la fel ca la prima optiune
- functionalitatea acestei optiuni fiind reprezentata intr-un al doilea rezultat extras in folder, "extracted_watermark.jpg", care va insera, mai intai, 
- peste imaginea originala, cea pentru watermark, rezultatul fiind salvat drept "result_image.jpg" iar din acest "result_image.jpg", se "decodeaza"
- informatia ascunsa peste imaginea originala (imaginea watermark)

"""


import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def add_watermark(image, text, opacity, rotation, position, margin, text_color):
    
    watermark = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8) #pentru initializarea imaginii transparente ce peste care aplicam textul, imagine transparenta ce
    #se va aplica peste imaginea noastra originala

    font = cv2.FONT_HERSHEY_SIMPLEX #selectam un font, marimea si grosimea
    font_scale = 1
    font_thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0] #calculam dimensiunea textului
    text_width, text_height = text_size

    if position == 'top': #in functie de alegerea user-ului, facem niste operatii specifice pentru coordonatele
        text_x = int((image.shape[1] - text_width) / 2) #unde urmeaza sa fie plasat initial watermark-ul
        text_y = margin + text_height
    elif position == 'bottom':
        text_x = int((image.shape[1] - text_width) / 2)
        text_y = image.shape[0] - margin
    elif position == 'left':
        text_x = margin
        text_y = int(image.shape[0] / 2 + text_height / 2)
    elif position == 'right':
        text_x = image.shape[1] - text_width - margin
        text_y = int(image.shape[0] / 2 + text_height / 2)
    elif position == 'center':
        text_x = int((image.shape[1] - text_width) / 2)
        text_y = int(image.shape[0] / 2 + text_height / 2)
    else:
        raise ValueError("Invalid position. Choose from 'top', 'bottom', 'left', 'right', 'center'.")

    if text_color == 'red': #in functie de alegerea user-ului, setam culoarea textului in formatul RGB, nealterate culorile (rosu simplu, albastru simplu etc)
        color = (0, 0, 255)
    elif text_color == 'blue':
        color = (255, 0, 0)
    elif text_color == 'white':
        color = (255, 255, 255)
    elif text_color == 'black':
        color = (0, 0, 0)
    elif text_color == 'yellow':
        color = (0, 255, 255)
    elif text_color == 'green':
        color = (0, 255, 0)
    else:
        raise ValueError("Invalid color. Choose from 'red', 'blue', 'white', 'black', 'yellow', 'green'.")

    cv2.putText(watermark, text, (text_x, text_y), font, font_scale, (*color, 255), font_thickness, cv2.LINE_AA) #punem textul propriu-zis peste imaginea transparenta
    
    rotation_matrix = cv2.getRotationMatrix2D((text_x + text_width // 2, text_y - text_height // 2), rotation, 1) #operatia de rotire daca user-ul a
    watermark = cv2.warpAffine(watermark, rotation_matrix, (watermark.shape[1], watermark.shape[0])) #ales aceasta functionabilitate

    alpha_channel = watermark[:, :, 3] / 255.0 * opacity #aici ajustam opacitatea in functie de alegerea userului, urmand mai jos sa combinam cele 2 imagini (cea originala si cea transparenta ce include watermark-ul)
    result = (image * (1.0 - alpha_channel[:, :, np.newaxis])).astype(np.uint8) + (watermark[:, :, :3] * alpha_channel[:, :, np.newaxis]).astype(np.uint8)

    return result

def add_lsb_watermark(image, watermark_image):
  
    if len(watermark_image.shape) > 2:
        watermark_image = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY) #aici transformam imaginea in format grayscale 

    watermark_resized = cv2.resize(watermark_image, (image.shape[1], image.shape[0])) #redimensionam imaginea ce urmeaza sa fie watermark
    #pentru a se potrivii cu dimensiunile imaginii peste care urmeaza sa aplicam acest watermark

    _, watermark_binary = cv2.threshold(watermark_resized, 127, 1, cv2.THRESH_BINARY) #transformarea la binar, cum este specificat in documentatie

    watermarked_image = np.copy(image)
    for i in range(3):  
        watermarked_image[:, :, i] = (watermarked_image[:, :, i] & 0xFE) | watermark_binary #pentru fiecare canal de culoare al imaginii originale,
        #aplicam watermark-ul pe LSB

    return watermarked_image

def extract_lsb_watermark(watermarked_image):
    
    lsb = np.bitwise_and(watermarked_image, 1) #extragem bitul LSB din fiecare pixel (cel mai din dreapta dintr-un byte)
   
    watermark_extracted = lsb[:, :, 0] * 255   #cream o imagine pentru watermark-ul extras
    
    watermark_equalized = cv2.equalizeHist(watermark_extracted) #egalizam histograma pentru o imbunatatire a esteticii rezultatului dorit
    
    _, watermark_threshold = cv2.threshold(watermark_equalized, 150, 255, cv2.THRESH_BINARY) #binarizam pentru a obtine watermark-ul final extractat

    return watermark_threshold

def calculate_entropy(signal):
    
    lensig = signal.size #functie pentru a calcula entropia
    symset = set(signal)
    propab = [np.size(signal[signal == i]) / lensig for i in symset]
    return -np.sum([p * np.log2(p) for p in propab if p > 0])

def local_entropy(image, neighborhood_size):
    
    rows, cols = image.shape #functie pentru calcularea entropiei pe fiecare pixel
    entropy_image = np.zeros_like(image, dtype=float)
    
    for row in range(rows): #parcugem fiecare pixel
        for col in range(cols):
            
            Lx, Ux = max(0, col - neighborhood_size), min(cols, col + neighborhood_size) #calculam entropia in vecinatatea fiecarui pixel
            Ly, Uy = max(0, row - neighborhood_size), min(rows, row + neighborhood_size)
            
            region = image[Ly:Uy, Lx:Ux].flatten() #calculeaza entropia pe regiunea selectata
            entropy_image[row, col] = calculate_entropy(region)

    return entropy_image

def main():
    
    choice = input("Choose watermarking method: 1 for Text Watermark, 2 for LSB Watermark: ") #aici utilizatorul introduce de la tastatura 
    #metoda de watermarking dorita: text sau LSB (Least Significant Bit)

    color_image = Image.open('psProject/test.jpg') #incarcam imaginea originala si o transformam in grayscale, iar "test.jpg" este imaginea peste care aplicam watermark-ul
    grey_image = color_image.convert('L')
    color_array = np.array(color_image)
    grey_array = np.array(grey_image)

    neighborhood_size = 5 #setam marimea vecinatatii (pentru calculul entropiei)
 
    entropy_map = local_entropy(grey_array, neighborhood_size) #aici calculam entropia locala si cea totala (cum am mai mentionat, entropie = masura nivelului de informatie/dezordinea/zgomotul imaginii)
    overall_entropy = calculate_entropy(grey_array.flatten())
    print(f"Overall Entropy of the grayscale image: {overall_entropy}")

    plt.figure(figsize=(15, 5)) #afisam folosind Matplotlib, un grafic cu trei imagini: originala, imaginea in tonuri alb-negru, entropia, si salvam
    plt.subplot(1, 3, 1) #entropia drept pdf
    plt.imshow(color_array)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(grey_array, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(1, 3, 3)
    plt.imshow(entropy_map, cmap='jet')
    plt.title('Local Entropy')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("Entropy.pdf", format='pdf')
    plt.show()

    if choice == '1': #aici este procesata optiunea aleasa de catre utilizator privind tehnica de aplicare a watermark-ului
        
        image_path = "psProject/test.jpg"
        watermark_text = input("Enter the watermark text: ") #pentru optiunea 1, utilizatorul trebuie sa introduca aceste optiuni pentru o personalizare proprie a watermark-ului
        opacity = float(input("Enter the opacity (0 to 1): "))
        rotation = float(input("Enter the rotation angle (in degrees): "))
        position = input("Enter the watermark position (top, bottom, left, right, center): ")
        margin = int(input("Enter the margin (in pixels) for the watermark position: "))
        text_color = input("Enter the text color (red, blue, white, black, yellow, green): ")

        image = cv2.imread(image_path)
        result = add_watermark(image, watermark_text, opacity, rotation, position, margin, text_color)

    elif choice == '2':
        
        image_path = "psProject/test.jpg" #pentru optiunea 2, LSB Watermarking, se aplica watermark-ul folosind tehnica LSB
        watermark_path = "psProject/test2.jpg" #urmand, dupa salvarea rezultatului inserarii watermark-ului peste imaginea originala, extragerea acestuia
        #pentru o verificare a corectitudinii

        image = cv2.imread(image_path)
        watermark_image = cv2.imread(watermark_path)

        result = add_lsb_watermark(image, watermark_image)
    else:
        print("Invalid choice.")
        return

    output_path = "psProject/result_image.jpg" #se salveaza imaginea cu acest nume si se afiseaza calea
    cv2.imwrite(output_path, result)
    print(f"Watermarked image saved at: {output_path}")

    if choice == '2': #aici utilizatorul este intrebat daca vrea sa extraga watermark-ul din imagine
        if input("Extract watermark? (y/n): ").lower() == 'y':

            watermarked_image_path = "psProject/result_image.jpg"
            watermarked_image = cv2.imread(watermarked_image_path)

            watermark = extract_lsb_watermark(watermarked_image)

            output_path = "psProject/extracted_watermark.jpg"
            cv2.imwrite(output_path, watermark)
            print(f"Extracted watermark image saved at: {output_path}")

if __name__ == "__main__":
    main()