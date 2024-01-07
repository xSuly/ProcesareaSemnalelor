# Tema 1
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
from scipy.fftpack import dct, idct
from skimage.color import rgb2ycbcr, ycbcr2rgb
import imageio
import cv2
import os

#la subpct 2 afisam si subpct 1

X = misc.ascent()

X2 = imageio.imread('https://github.com/scikit-image/scikit-image/raw/main/skimage/data/astronaut.png') #o imagine RGB cu un astronaut din biblioteca skimage
plt.imshow(X2, cmap=plt.cm.gray)
plt.title('imaginea rgb')
plt.show()

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

blocuri_originale = []
blocuri_jpeg = []


#subpct 1
for i in range(0, X.shape[0], 8): #imaginea noastra X, are 512x512 deci trebuie sa parcurgem fiecare bloc 8x8
    for j in range(0, X.shape[1], 8):
        block = X[i:i+8, j:j+8] #aplicam codul din cerinte doar ca pe fiecare bloc 8x8
        y = dctn(block)
        y_jpeg = Q_jpeg * np.round(y/Q_jpeg)
        x_jpeg = idctn(y_jpeg)
        blocuri_originale.append(block)
        blocuri_jpeg.append(x_jpeg)

#subpct 2
        
YCbCr_image = rgb2ycbcr(X2)
YCbCr_image_afisare = np.clip(YCbCr_image, 0, 255).astype(np.uint8) #luam imaginea noastra RGB si o transformam in YCbCr, dupa verificam ca fiecare valoare
plt.imshow(YCbCr_image_afisare) #din imainea noastra are valori intre 0 si 255 (pixelii reprezentati in 8 biti) iar daca gaseste valori mai mici decat 0
plt.title('YCbCr Image') #se seteaza la 0 iar daca sunt mai mari decat 255, se seteaza la 255
plt.show()

matrice_CbCr = ([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]]) #matricea Q_Jpeg o folosim pentru componenta Y (lumina) iar aceasta matrice pentru
#componentele Cb si Cr (componentele pentru albastru si rosu)

canalul_Y = [] #folosim pentru a face append sa vedem cum functioneaza exact codul si pt a afisa in final concatenarea corecta
canalul_Cb = []
canalul_Cr = []

for canal in range(YCbCr_image.shape[2]): #facem operatiile de comprimare DCT pe fiecare canal iar la final le unim pe toate cele 3 canale
    if canal == 0:
        Q_jpeg_folosit = Q_jpeg
    else:
        Q_jpeg_folosit = matrice_CbCr
    blocuri_comprimate = []
    for i in range(0, YCbCr_image.shape[0], 8): 
        for j in range(0, YCbCr_image.shape[1], 8):
            block = YCbCr_image[i:i+8, j:j+8, canal]  
            y = dctn(block)  
            y_jpeg = np.round(y / Q_jpeg_folosit) * Q_jpeg_folosit  
            x_jpeg = idctn(y_jpeg)
            blocuri_comprimate.append(x_jpeg)

    if canal == 0:
        canalul_Y = blocuri_comprimate
    if canal == 1:
        canalul_Cb = blocuri_comprimate
    if canal == 2:
        canalul_Cr = blocuri_comprimate #adaugam fiecare componenta in matricea sa

imagine_comprimata_final = np.stack([np.block([canalul_Y[i : i + 8*8] for i in range(0, len(canalul_Y), 8*8)]),
                                      np.block([canalul_Cb[i : i + 8*8] for i in range(0, len(canalul_Cb), 8*8)]),
                                        np.block([canalul_Cr[i : i + 8*8] for i in range(0, len(canalul_Cr), 8*8)])], axis=2)



#################################

plt.subplot(2, 1, 1)
concatenated_image_original = np.concatenate([np.concatenate(row, axis=1) for row in np.array_split(blocuri_originale, X.shape[0]//8)], axis=0)
plt.imshow(concatenated_image_original, cmap = 'grey') #np.array_split impartea matricea noastra intr-un array de submatrici ca nr luand dimensiunea(x) din X impartita la 8 (deci 512 / 8) cu valori 8x8
plt.title('Blocuri Originale') #apoi se face concatenate pe fiecare rand sub forma unei liste iar mai apoi lista se concateneaza pe axa 0 (adica vertical)


plt.subplot(2, 1, 2)
concatenated_image_jpeg = np.concatenate([np.concatenate(row, axis=1) for row in np.array_split(blocuri_jpeg, X.shape[0]//8)], axis=0)
plt.imshow(concatenated_image_jpeg, cmap = 'grey')
plt.title('Blocuri JPEG')

plt.tight_layout()
plt.show()


plt.imshow(ycbcr2rgb(imagine_comprimata_final))
plt.title('comprimare YCbCr JPEG')
plt.show()

#subpct 3
def calculeaza_mse(original, compressed):
    return np.mean((original - compressed) ** 2) #formula pentru mean_squared_error


mse_utilizator = 100
Y = misc.ascent()
compressed_image = np.zeros_like(Y)

while True:
    for i in range(0, Y.shape[0], 8):
        for j in range(0, Y.shape[1], 8):
            block = Y[i:i+8, j:j+8]
            y = dctn(block)
            y_jpeg = block * np.round(y/block)
            x_jpeg = idctn(y_jpeg)
            compressed_image[i:i+8, j:j+8] = x_jpeg #actualizam imaginea comprimata pentru fiecare bloc

    mse = calculeaza_mse(Y, compressed_image)
    print(f"MSE: {mse}")

    if mse <= mse_utilizator:
        break


#subpct 4


video_path = 'Tema1/cai.mp4' #cale catre un video

cap = cv2.VideoCapture(video_path)
output_directory = 'Tema1/output_video/' #calea unde salvam cadrele noastre din video
os.makedirs(output_directory, exist_ok=True) #creare director daca nu exista

fps = cap.get(cv2.CAP_PROP_FPS) #cum videoclipul are 30fps (30 de cadre pe secunda) vreau sa afisez doar primul cadru dintr-o secunda, nu toate 30

frame_count = 0
seconds_count = 0
iterare_salvare_poze = 0
while True:
    
    ret, frame = cap.read() #citim cadrele din video
    if not ret: #verificam sa nu fim la final, daca am ajuns la final iesim din bucla while True
        break

    frame_count += 1
    iterare_salvare_poze +=1 #o sa se salveze cu nr ciudate pentru ca luam primul cadru din cele 24 sau 30 cadre per secunda, cate are videoclipul
    if frame_count > fps:
        frame_count = 0
        seconds_count += 1
        output_path = f'{output_directory}/frame_{iterare_salvare_poze:04d}.jpg' #salvam cadrele
        cv2.imwrite(output_path, frame)
        YCbCr_frame = rgb2ycbcr(frame)
        YCbCr_image_afisare2 = np.clip(YCbCr_frame, 0, 255).astype(np.uint8)

        canalul_Y_fps = []
        canalul_Cb_fps = []
        canalul_Cr_fps = []
        
        for canal in range(YCbCr_frame.shape[2]): 
            if canal == 0:
                Q_jpeg_folosit = Q_jpeg
            else:
                Q_jpeg_folosit = matrice_CbCr
            blocuri_comprimate_fps = []
            for i in range(0, YCbCr_frame.shape[0], 8): 
                for j in range(0, YCbCr_frame.shape[1], 8):
                    block = YCbCr_frame[i:i+8, j:j+8, canal]  
                    y = dctn(block)  
                    y_jpeg = np.round(y / Q_jpeg_folosit) * Q_jpeg_folosit  
                    x_jpeg = idctn(y_jpeg)
                    blocuri_comprimate_fps.append(x_jpeg)

            if canal == 0:
                canalul_Y_fps = blocuri_comprimate_fps
            if canal == 1:
                canalul_Cb_fps = blocuri_comprimate_fps
            if canal == 2:
                canalul_Cr_fps = blocuri_comprimate_fps #reutilizare subpct2

        size = YCbCr_frame.shape[1] // 8
        imagine_comprimata_final_fps = np.stack([
            np.block([canalul_Y_fps[i : i + size] for i in range(0, len(canalul_Y_fps), size)]),
            np.block([canalul_Cb_fps[i : i + size] for i in range(0, len(canalul_Cb_fps), size)]),
            np.block([canalul_Cr_fps[i : i + size] for i in range(0, len(canalul_Cr_fps), size)])
        ], axis=2)

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].imshow(frame)
        axs[0].set_title('Cadru original RGB')

        axs[1].imshow(YCbCr_image_afisare2)
        axs[1].set_title('YCbCr')

        axs[2].imshow(ycbcr2rgb(imagine_comprimata_final_fps))
        axs[2].set_title('Restaurare din YCbCr')

        plt.tight_layout()
        plt.show()

cap.release() #eliberam resursele
cv2.destroyAllWindows()




