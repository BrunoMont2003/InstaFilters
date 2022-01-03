import tkinter as tk
from tkinter.constants import DISABLED
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np
import dlib
from math import hypot


ventana = tk.Tk()

ventana.geometry("428x926+200+10")
ventana.title("Filtros de Instagram")
ventana.resizable(width=False, height=False)


altura = 640
anchura = 640
# altura = 926
# anchura = 428

# GLOBALES
video = None
capturar = None
img_counter = 0
salida = None
grabar = False
IP = "http://192.168.0.12:8080/video"


def video_stream():
    global video
    # video = cv2.VideoCapture(0)
    video = cv2.VideoCapture(IP)
    iniciar(video)


def iniciar(captura):
    global video
    ret, frame = captura.read()
    if ret == True:
        frame = imutils.resize(frame, height=altura)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=img)
        etiq_video.configure(image=image)
        etiq_video.image = image
        etiq_video.after(10, iniciar, captura)
    else:
        etiq_video.image = ""
        captura.release()


def quitar():
    global video
    # etiq_video.pack_forget()
    video.release()


def grabarVideo(captura, salida, contador, duracion):
    etiq_video.after(10, quitar)
    #global video, grabar
    ret, frame = captura.read()
    if ret and contador < duracion:
        salida.write(frame)
        frame = imutils.resize(frame, height=altura)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=img)
        etiq_video.configure(image=image)
        etiq_video.image = image
        print(contador)
        etiq_video.after(10, grabarVideo, captura,
                         salida, contador+1, duracion)
    else:
        captura.release()
        etiq_video.after(10, video_stream)


def detenerVideo():
    global salida, grabar
    if salida is not None:
        grabar = False
        salida.release()


def recording():
    etiq_video.after(10, quitar)

    salida = cv2.VideoWriter(
        'records/videoSalida.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (480, 640))

    captura = cv2.VideoCapture(IP)
    print(captura.get(cv2.CAP_PROP_FRAME_WIDTH), "x",
          captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grabarVideo(captura, salida, 0, 100)


def webcamToggle():
    global bandera
    if bandera:
        video_stream()
        bandera = False

    else:
        quitar()
        bandera = True


def tomarFoto():
    global video, img_counter
    control = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("failed to grab frame")
            break
        if control == 0:
            img_name = "capturas/foto_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            control += 1
        else:
            break


def llamarNewYear():
    cap = cv2.VideoCapture(IP)

    image = cv2.imread('img/nuevo1.png', cv2.IMREAD_UNCHANGED)

    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    newyear(cap, image, faceClassif)


def newyear(cap, image, faceClassif):
    etiq_video.after(10, quitar)

    ret, frame = cap.read()
    if ret:
        faces = faceClassif.detectMultiScale(frame, 1.3, 5)

        contador = 0
        for (x, y, w, h) in faces:
            resized_image = imutils.resize(image, width=w)
            filas_image = resized_image.shape[0]
            col_image = w
            porcion_alto = filas_image // 4
            dif = 0
            if y + porcion_alto - filas_image >= 0:
                n_frame = frame[y + porcion_alto - filas_image: y + porcion_alto,
                                x: x + col_image]
            else:
                dif = abs(y + porcion_alto - filas_image)
                n_frame = frame[0: y + porcion_alto,
                                x: x + col_image]

            mask = resized_image[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)
            bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
            bg_black = bg_black[dif:, :, 0:3]
            bg_frame = cv2.bitwise_and(
                n_frame, n_frame, mask=mask_inv[dif:, :])

            result = cv2.add(bg_black, bg_frame)
            if y + porcion_alto - filas_image >= 0:
                frame[y + porcion_alto - filas_image: y +
                      porcion_alto, x: x + col_image] = result

            else:
                frame[0: y + porcion_alto, x: x + col_image] = result
            contador += 1
            print(contador)
        frame = imutils.resize(frame, height=altura)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imagen = ImageTk.PhotoImage(image=img)
        etiq_video.configure(image=imagen)
        etiq_video.image = imagen
        frame = imutils.resize(frame, width=640)
        etiq_video.after(10, newyear, cap, image, faceClassif)
    else:
        cap.release()
        etiq_video.after(10, video_stream)


def llamarCachos():
    cap = cv2.VideoCapture(IP)
    # Lectura de la imagen a incrustar en el video
    image = cv2.imread('img/cachos.png', cv2.IMREAD_UNCHANGED)

    # Detector de rostros
    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    newyear(cap, image, faceClassif)


def caracachos(cap, image, faceClassif):
    etiq_video.after(10, quitar)

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    if ret:

        faces = faceClassif.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            resized_image = imutils.resize(image, width=w)
            filas_image = resized_image.shape[0]
            col_image = w
            porcion_alto = filas_image // 4
            dif = 0
            if y + porcion_alto - filas_image >= 0:

                n_frame = frame[y + porcion_alto - filas_image: y + porcion_alto,
                                x: x + col_image]
            else:
                dif = abs(y + porcion_alto - filas_image)
                n_frame = frame[0: y + porcion_alto,
                                x: x + col_image]

            mask = resized_image[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)

            bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
            bg_black = bg_black[dif:, :, 0:3]
            bg_frame = cv2.bitwise_and(
                n_frame, n_frame, mask=mask_inv[dif:, :])

            result = cv2.add(bg_black, bg_frame)
            if y + porcion_alto - filas_image >= 0:
                frame[y + porcion_alto - filas_image: y +
                      porcion_alto, x: x + col_image] = result

            else:
                frame[0: y + porcion_alto, x: x + col_image] = result

        frame = imutils.resize(frame, height=altura)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imagen = ImageTk.PhotoImage(image=img)
        etiq_video.configure(image=imagen)
        etiq_video.image = imagen
        frame = imutils.resize(frame, width=640)
        etiq_video.after(10, caracachos, cap, image, faceClassif)


def llamarCerdo():
    cap = cv2.VideoCapture(IP)
    nose_image = cv2.imread('img/pig_nose.png')
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

    # Detector facial de carga
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'carga/shape_predictor_68_face_landmarks.dat')
    cerdo(cap, nose_image, nose_mask, detector, predictor)


def cerdo(cap, nose_image, nose_mask, detector, predictor):
    etiq_video.after(10, quitar)
    ret, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:

        faces = detector(frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

            # Coordenadas de la nariz
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                                   left_nose[1] - right_nose[1]) * 1.7)
            nose_height = int(nose_width * 0.77)

            # Nueva posición de la nariz
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Añadiendo la nueva nariz
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(
                nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                              top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(
                nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]: top_left[1] + nose_height,
                  top_left[0]: top_left[0] + nose_width] = final_nose

    frame = imutils.resize(frame, height=altura)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imagen = ImageTk.PhotoImage(image=img)
    etiq_video.configure(image=imagen)
    etiq_video.image = imagen
    frame = imutils.resize(frame, width=640)
    etiq_video.after(10, cerdo, cap, nose_image,
                     nose_mask, detector, predictor)


def mostrarIconosFoto():
    panel_video.pack_forget()
    shot_boton.pack()


def mostrarIconosVideo():
    panel_video.pack()
    shot_boton.pack_forget()


# etiqueta
etiq_video = tk.Label(ventana, bg="black")
etiq_video.place(x=0, y=0)

bandera = True
webcamToggle()
# on off
onoff_icon = tk.PhotoImage(file="img/onoff.png")

onoff_boton = tk.Button(ventana, image=onoff_icon, cursor="hand2",
                        border=0, command=webcamToggle)
onoff_boton.place(x=10, y=10)

# panel container
container = tk.LabelFrame(ventana, height=75, width=anchura, bd=0)
container.place(x=140, y=715)

# boton capturar imagen
icon = tk.PhotoImage(file="img/shot.png")
shot_boton = tk.Button(container, image=icon, cursor="hand2",
                       border=0, width=150, command=tomarFoto)
shot_boton.pack(side="left")


# botones capturar video
panel_video = tk.LabelFrame(container, height=75, width=anchura, bd=0)


record_icon = tk.PhotoImage(file="img/record.png")
record_boton = tk.Button(panel_video, image=record_icon, cursor="hand2",
                         border=0, command=lambda: recording()).grid(row=0, column=1, padx=12)

stop_icon = tk.PhotoImage(file="img/stop.png")
stop_boton = tk.Button(panel_video, image=stop_icon, cursor="hand2",
                       border=0, command=lambda: recording()).grid(row=0, column=0, padx=12)


# escoger modo
panel_modo = tk.LabelFrame(ventana, height=50, width=anchura, bd=0)
panel_modo.place(x=123, y=800)
imagen_icon = tk.PhotoImage(file="img/btnImagen.png")
imagen_boton = tk.Button(panel_modo, image=imagen_icon,
                         cursor="hand2", border=0, command=mostrarIconosFoto).grid(row=0, column=0, padx=10)
video_icon = tk.PhotoImage(file="img/btnVideo.png")
video_boton = tk.Button(panel_modo, image=video_icon,
                        cursor="hand2", border=0, command=mostrarIconosVideo).grid(row=0, column=1, padx=10)


#  panel de filtros
panel = tk.LabelFrame(ventana, height=85,
                      width=anchura, bd=0)
panel.place(x=0, y=842)

cuadrado = tk.PhotoImage(file="img/Rectangle.png")
item1 = tk.Button(panel, image=cuadrado, cursor="hand2",
                  border=0, command=llamarNewYear).grid(row=0, column=0, padx=13)
item2 = tk.Button(panel, image=cuadrado, cursor="hand2",
                  border=0, command=llamarCachos).grid(row=0, column=1, padx=13)
item3 = tk.Button(panel, image=cuadrado, cursor="hand2",
                  border=0, command=llamarCerdo).grid(row=0, column=2, padx=13)
item4 = tk.Button(panel, image=cuadrado, cursor="hand2",
                  border=0).grid(row=0, column=3, padx=13)
item5 = tk.Button(panel, image=cuadrado, cursor="hand2",
                  border=0).grid(row=0, column=4, padx=13)


ventana.mainloop()
