import tkinter as tk
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np
import dlib
from math import hypot


ventana = tk.Tk()
ventana.geometry("428x926+2000+10")
ventana.title("Filtros de Instagram")
ventana.resizable(width=False, height=False)


# GLOBALES
altura = 640
anchura = 428
video = None
img_counter = 0
salida = None
IP = "http://192.168.0.12:8080/video"
ayuda_ny = 0
ayuda_c = 0
ayuda_nc = 0
ayuda_mc = 0
video_ny = None
video_c = None
video_nc = None
video_mc = None


def video_stream():
    global video
    video = cv2.VideoCapture(IP)
    # video = cv2.VideoCapture(0)
    iniciar(video)


def iniciar(captura):
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
    video.release()


def reset():
    global video_ny, video_c, video_nc, video_mc
    if video_ny is not None:
        video_ny.release()
    if video_c is not None:
        video_c.release()
    if video_nc is not None:
        video_nc.release()
    if video_mc is not None:
        video_mc.release()


def recording():
    global video
    captura = video
    salida = cv2.VideoWriter(
        'records/videoSalida.avi', cv2.
        VideoWriter_fourcc(*'XVID'), 20.0, (480, 640))
    grabarVideo(captura, salida, 0, 100)


def grabarVideo(captura, salida, contador, duracion):
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


def webcamToggle():
    global bandera
    if bandera:
        video_stream()
        bandera = False

    else:
        quitar()
        bandera = True


def tomarFoto(cap):
    global video, img_counter
    control = 0

    while True:
        ret, frame = cap.read()
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
    reset()
    global ayuda_ny, video_ny
    ayuda_ny += 1
    if ayuda_ny == 1:
        video_ny = cv2.VideoCapture(IP)


def llamarTomarFoto():
    global video, ayuda_ny, ayuda_c, ayuda_nc, ayuda_mc
    if ayuda_ny == 1:
        llamarNewYear(True)
        ayuda_ny = 0
    if ayuda_c == 1:
        llamarCachos(True)
    if ayuda_nc == 1:
        llamarCerdo(True)
    if ayuda_mc == 1:
        llamarMclovin(True)
    else:
        tomarFoto(video)


def llamarNewYear(shot):
    reset()
    global ayuda_ny, video_ny
    if shot:
        ayuda_ny = 0
    ayuda_ny += 1
    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread('img/nuevo1.png', cv2.IMREAD_UNCHANGED)

    if not shot:
        if ayuda_ny == 1:
            video_ny = cv2.VideoCapture(IP)

            newyear(video_ny, image, faceClassif, False)
        else:
            reset()
            quitar()
            print("god")
            ayuda_ny = 0
    else:
        video_ny = cv2.VideoCapture(IP)
        print("shot true")
        print(video_ny.get(cv2.CAP_PROP_FRAME_WIDTH))
        newyear(video_ny, image, faceClassif, True)


contador = 0


def newyear(cap, image, faceClassif, shot):
    global ayuda_ny, img_counter
    etiq_video.after(10, quitar)
    ret, frame = cap.read()
    if ret and ayuda_ny == 1:
        # print(contador+1)
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
        if shot:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_name = "capturas/foto_filtro_ny_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            shot = False
        etiq_video.after(10, newyear, cap, image, faceClassif, shot)
    else:
        cap.release()
        etiq_video.after(10, video_stream)
        print("el ret murio", "ret: ", ret, "ayuda: ", ayuda_ny)
        reset()


def llamarCachos(shot):
    reset()
    global ayuda_c, video_c
    if shot:
        ayuda_c = 0
    ayuda_c += 1
    image = cv2.imread('img/cachos.png', cv2.IMREAD_UNCHANGED)
    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not shot:
        if ayuda_c == 1:
            video_c = cv2.VideoCapture(IP)
            caracachos(video_c, image, faceClassif, False)

        else:
            quitar()
            print("god")
            ayuda_c = 0
            reset()
    else:
        video_c = cv2.VideoCapture(IP)
        caracachos(video_c, image, faceClassif, True)


def caracachos(cap, image, faceClassif, shot):
    global ayuda_c, img_counter
    etiq_video.after(10, quitar)

    ret, frame = cap.read()
    if ret and ayuda_c == 1:
        frame = imutils.resize(frame, width=640)

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
        if shot:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_name = "capturas/foto_filtro_cuernos_{}.png".format(
                img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            shot = False
        etiq_video.after(10, caracachos, cap, image, faceClassif, shot)

    else:
        cap.release()
        etiq_video.after(10, video_stream)
        reset()
        print("el ret murio")


def llamarCerdo(shot):
    reset()
    global ayuda_nc, video_nc
    if shot:
        ayuda_nc = 0
    ayuda_nc += 1
    nose_image = cv2.imread('img/pig_nose.png')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'carga/shape_predictor_68_face_landmarks.dat')

    if not shot:
        if ayuda_nc == 1:
            video_nc = cv2.VideoCapture(IP)
            ret, frame = video_nc.read()
            if ret:
                rows, cols, _ = frame.shape
                nose_mask = np.zeros((rows, cols), np.uint8)
            cerdo(video_nc, nose_image, nose_mask, detector, predictor, False)
        else:
            quitar()
            print('god')
            ayuda_nc = 0
            reset()
    else:
        video_nc = cv2.VideoCapture(IP)
        ret, frame = video_nc.read()
        if ret:
            rows, cols, _ = frame.shape
            nose_mask = np.zeros((rows, cols), np.uint8)
        cerdo(video_nc, nose_image, nose_mask, detector, predictor, True)


def cerdo(cap, nose_image, nose_mask, detector, predictor, shot):
    global ayuda_nc, img_counter
    etiq_video.after(10, quitar)
    ret, frame = cap.read()
    nose_mask.fill(0)

    if ret and ayuda_nc == 1:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            # Nueva posici??n de la nariz
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # A??adiendo la nueva nariz
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
        if shot:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            img_name = "capturas/foto_filtro_narizcerdo_{}.png".format(
                img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            shot = False
        etiq_video.after(10, cerdo, cap, nose_image,
                         nose_mask, detector, predictor, shot)
    else:
        cap.release()
        print("el ret murio")
        reset()
        etiq_video.after(10, video_stream)


def llamarMclovin(shot):
    reset()
    global ayuda_mc, video_mc
    if shot:
        ayuda_mc = 0
    ayuda_mc += 1

    image = cv2.imread('img/mclovin.png', cv2.IMREAD_UNCHANGED)
    faceClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not shot:
        if ayuda_mc == 1:
            video_mc = cv2.VideoCapture(IP)
            mclovin(video_mc, image, faceClassif, False)
        else:
            print("god")
            ayuda_mc = 0
            reset()
    else:
        video_mc = cv2.VideoCapture(IP)
        mclovin(video_mc, image, faceClassif, True)


def mclovin(cap, image, faceClassif, shot):
    global ayuda_mc, img_counter
    etiq_video.after(10, quitar)
    ret, frame = cap.read()
    if ret and ayuda_mc == 1:
        frame = imutils.resize(frame, width=640)
        faces = faceClassif.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:

            resized_image = imutils.resize(image, width=w)
            filas_image = resized_image.shape[0]
            col_image = w

            porcion_alto = filas_image // 2

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
        if shot:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_name = "capturas/foto_filtro_mclovin_{}.png".format(
                img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            shot = False
        etiq_video.after(10, mclovin, cap, image, faceClassif, shot)
    else:
        cap.release()
        etiq_video.after(10, video_stream)
        print("el ret murio")
        reset()


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
onoff_boton = tk.Button(ventana,
                        image=onoff_icon,
                        cursor="hand2",
                        border=0,
                        command=webcamToggle)
onoff_boton.place(x=10, y=10)

# panel container
container = tk.LabelFrame(ventana,
                          height=75,
                          width=anchura,
                          bd=0)
container.place(x=140, y=715)

# boton capturar imagen
icon = tk.PhotoImage(file="img/shot.png")
shot_boton = tk.Button(container,
                       image=icon,
                       cursor="hand2",
                       border=0,
                       width=150,
                       command=llamarTomarFoto)
shot_boton.pack(side="left")


# botones capturar video
panel_video = tk.LabelFrame(container, height=75,
                            width=anchura, bd=0)
record_icon = tk.PhotoImage(file="img/record.png")
record_boton = tk.Button(panel_video, image=record_icon,
                         cursor="hand2",
                         border=0,
                         command=recording).grid(row=0,
                                                 column=1,
                                                 padx=12)

stop_icon = tk.PhotoImage(file="img/stop.png")
stop_boton = tk.Button(panel_video, image=stop_icon, cursor="hand2",
                       border=0).grid(row=0, column=0, padx=12)


# escoger modo
panel_modo = tk.LabelFrame(ventana, height=50,
                           width=anchura, bd=0)
panel_modo.place(x=123, y=800)
imagen_icon = tk.PhotoImage(file="img/btnImagen.png")
imagen_boton = tk.Button(panel_modo, image=imagen_icon,
                         cursor="hand2", border=0,
                         command=mostrarIconosFoto).grid(row=0,
                                                         column=0,
                                                         padx=10)
video_icon = tk.PhotoImage(file="img/btnVideo.png")
video_boton = tk.Button(panel_modo, image=video_icon,
                        cursor="hand2", border=0,
                        command=mostrarIconosVideo).grid(row=0,
                                                         column=1,
                                                         padx=10)


#  panel de filtros
panel = tk.LabelFrame(ventana, height=85,
                      width=anchura, bd=0)
panel.place(x=0, y=842)

gorro = tk.PhotoImage(file="img/gorro.png")
cuernos = tk.PhotoImage(file="img/cuernos.png")
cerdo_i = tk.PhotoImage(file="img/cerdo.png")
mclovin_item = tk.PhotoImage(file="img/mclovin_item.png")

item1 = tk.Button(panel, image=gorro, cursor="hand2",
                  border=0,
                  command=lambda: llamarNewYear(False)).grid(row=0,
                                                             column=0,
                                                             padx=20)
item2 = tk.Button(panel, image=cuernos, cursor="hand2",
                  border=0,
                  command=lambda: llamarCachos(False)).grid(row=0,
                                                            column=1,
                                                            padx=20)
item3 = tk.Button(panel, image=cerdo_i, cursor="hand2",
                  border=0,
                  command=lambda: llamarCerdo(False)).grid(row=0,
                                                           column=2,
                                                           padx=20)
item4 = tk.Button(panel, image=mclovin_item, cursor="hand2",
                  border=0,
                  command=lambda: llamarMclovin(False)).grid(row=0,
                                                             column=3,
                                                             padx=20)

ventana.mainloop()
