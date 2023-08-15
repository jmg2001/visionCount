import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from pyModbusTCP.client import ModbusClient
import time
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import time, logging
import threading
import subprocess
import socket
from datetime import datetime

global no_detecting
no_detecting = True

# ----------------- Funcion para guardar los datos en el csv ----------------- #
import csv

nombre_archivo = 'datos.csv'

def agregar_datos(datos):
    with open(nombre_archivo, mode='a', newline='') as archivo:
        escritor_csv = csv.writer(archivo)
        escritor_csv.writerow(datos)
        archivo.close()
    print("Datos agregados")

agregar_datos(["id","conteo","date","promedio"])

global data_counter

data_counter = 0
# ------------------------------------- . ------------------------------------ #

# ----------------------- Diccionario con los formatos ----------------------- #
formatos = {"penny":182,
            "hexgonal": 49}

formato_actual = "penny"
# ------------------------------------- . ------------------------------------ #

# --------------------- Funciones para menu de la ventana -------------------- #
def toggle_fullscreen(event=None):  # Funcion para cambiar de full screen
    root.attributes("-fullscreen", False)
    # return "break"

def toggle_fullscreen_on(event=None):  # Funcion para cambiar de full screen
    root.attributes("-fullscreen", True)
    # return "break"
# ------------------------------------- . ------------------------------------ #

# ------------------------ Configuracion de la ventana ----------------------- #
# Crea la ventana para GUI
root = tk.Tk()
root.wm_title("Automatización Deep detector")
# root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.config(background='#2b302c')
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.attributes("-fullscreen", False)  # Correr programa en full screen
root.bind("<Escape>", toggle_fullscreen)
root.bind("<F11>", toggle_fullscreen_on)
# ------------------------------------- . ------------------------------------ #


def setReinicia():
    os.system('sudo reboot')


# --------------------------- Frames de la ventana --------------------------- #
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Reboot", command=setReinicia)
menubar.add_cascade(label="File", menu=filemenu)

upFrame = Frame(root, width=width)
upFrame.config(background="#2b302c")
upFrame.grid_columnconfigure(0, weight=2)
upFrame.grid_rowconfigure(0, weight=3)
upFrame.grid(row=1, column=0, padx=10, pady=2, sticky='WENS')

leftFrame = Frame(upFrame)
leftFrame.grid_columnconfigure(0, weight=0)
leftFrame.grid(row=0, column=0, padx=20, pady=2, sticky="EW")

rightFrame = Frame(upFrame)
rightFrame.grid_columnconfigure(0, weight=0)
rightFrame.grid(row=0, column=1, padx=20, pady=2, sticky="EW")

global hex_verde, hex_rojo_, hex_amarillo
hex_verde = "#14FF09"
hex_amarillo = "#F4FF09"
hex_rojo = "#FF0909"

semaforoFrame = Frame(root, width=500, bg='white', height=20)
semaforoFrame.grid_columnconfigure(0, weight=1)
semaforoFrame.grid_rowconfigure(0, weight=1)
semaforoFrame.grid(row=0, column=0, padx=20, pady=2, columnspan=2, sticky='SWEN')

# Crea imagen y la coloca en pantalla
# try:
#     im = Image.open("opencv_frame_21.png")
#     im = im.resize((500, 400))
# except:
im = im = Image.open("base.jpg")
im = im.resize((500, 400))


photoOrig = ImageTk.PhotoImage(im)
labelFotoOrig = Label(leftFrame, image=photoOrig)
labelFotoOrig.grid_columnconfigure(0, weight=1)
labelFotoOrig.grid(row=0, column=0, sticky="NS")


photo = ImageTk.PhotoImage(im)
labelFoto = Label(rightFrame, image=photo)
labelFoto.grid_columnconfigure(0, weight=1)
labelFoto.grid(row=0, column=0, sticky="NS")


downFrame = Frame(root, height=height / 70)
downFrame.grid_rowconfigure(0, weight=1)
downFrame.grid_columnconfigure(0, weight=1)
downFrame.config(background="#FFFFFF")
downFrame.grid(row=2, column=0, columnspan=2, padx=10, pady=2, sticky='SWE')

scrollbar = Scrollbar(downFrame)
scrollbar.grid(row=0, column=1)
w = width
log = Text(downFrame, takefocus=0, height=height / 70)
log.grid(row=0, column=0, sticky='WE')

log.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=log.yview)
# ------------------------------------- . ------------------------------------ #

# ------------------------------ Funcion del log ----------------------------- #
def loginsert(mess):
    print(mess)
    #logger.info(mess)
    log.insert(END, mess + "\n")
    log.see(END)
# ------------------------------------- . ------------------------------------ #

# ----------------- Variables globales Trigger, Color, PerTot ---------------- #
global trigger
trigger = False
global color
color = 4
global perTot 
perTot = 0
global dataRecolected
dataRecolected = False
# ------------------------------------- . ------------------------------------ #


# ---------------------------- Carga de los pesos ---------------------------- #
weights = formato_actual + ".pt"                              
# Load model
model = attempt_load(weights, map_location="cpu")  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

if False:
    model = TracedModel(model, "cpu", 640)
# ------------------------------------- . ------------------------------------ #

# ----------------------------- Funcion detect() ----------------------------- #

def detect(file):
    print("----------------------------- Ejecutando Detect ----------------------------")
    global data_counter, no_detecting, color, perTot, dataRecolected, hex_verde, hex_amarillo, hex_rojo, formatos
    
    time_now = datetime.now()
    no_detecting = False
    detecting = False
    color_text = "No detectados"
    
    opt.source = file
    weights = formato_actual + ".pt" 
    opt.weights = formato_actual + ".pt" 
    opt.conf_thres = 0.45
    opt.img_size = 640
    opt.view_img = True

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #print("+++++++++++save_dir")
    #print(no_detecting)
    print(save_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    print("Device:")
    print(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections.
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)} "  # add to string
                    calcPer(int(n), 182, 10)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        #Se coloca el texto de cuantas piezas se detectaron en las imagenes
                        cv2.putText(im0,f"{n} {names[int(cls)]}", org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,255,0),thickness=3)

                        #Segun el color que se tenga se elige el color del texto del estado
                        if color == 1:
                            color_text = "Verde"
                            color_label = (0,255,0)
                            color_bar = hex_verde
                        if color == 2:
                            color_text = "Amarillo"
                            color_label = (0,255,255)
                            color_bar = hex_amarillo
                        if color == 3:
                            color_text = "Rojo"
                            color_label = (0,0,255)
                            color_bar = hex_rojo
                        if color == 4:
                            color_text = "NA"
                            color_label = (255,255,255)
                            color_bar = "#FFFFFF"
                        cv2.putText(im0,f"{color_text}", org=(500,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=color_label,thickness=3)
                        semaforoFrame.configure(bg=color_bar)
                        #Se crea la variable que indica que se detectaron objetos   
                        detecting = True
            
            #Si no se detectaron objetos se colocan textos default
            if not detecting:
                calcPer(0,182,10)
                if color == 1:
                    color_text = "Verde"
                    color_label = (0,255,0)
                    color_bar = hex_verde
                if color == 2:
                    color_text = "Amarillo"
                    color_label = (0,255,255)
                    color_bar = hex_amarillo
                if color == 3:
                    color_text = "Rojo"
                    color_label = (0,0,255)
                    color_bar = hex_rojo
                if color == 4:
                    color_text = "NA"
                    color_label = (255,255,255)
                    color_bar = "#FFFFFF"
                cv2.putText(im0,f"0 {formato_actual}", org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,255,0),thickness=3)
                cv2.putText(im0,f"{color_text}", org=(500,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=color_label,thickness=3)
                semaforoFrame.configure(bg=color_bar)

            #Se obtiene la eficiciencia y se le da formato
            porcentaje = perTot*100
            porcentaje = format(porcentaje, '.2f')

            #Si se detectaron objetos se coloca la cantidad y la eficiencia, si aun no se tienen 10 datos se coloca la leyenda correspondiente
            if detecting:
                if dataRecolected:
                    loginsert(s + ' - Eficiencia:' + str(porcentaje) + '%  --> ' + color_text + " || " + str(time_now))
                else:
                    loginsert(s + ' - ' + "Recolectando datos" + " || " + str(time_now))
            #Si no se detectaron objetos se mandan los valores default
            else:
                if dataRecolected:
                    loginsert('0 penny' + ' - Eficiencia:' + str(porcentaje) + '%  --> ' + color_text + " || " + str(time_now))
                else:
                    loginsert('0 penny' + ' - ' + "Recolectando datos" + " || " + str(time_now))

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            view_img = False
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(20)  # 1 millisecond
            
            #Si se detectaron obejtos se guardan los datos
            if len(s.split(' ')) > 1:
                string = s.split(' ')
                string = string[0]
                if dataRecolected:
                    agregar_datos([data_counter, string, time_now, porcentaje])
                else:
                    agregar_datos([data_counter, string, time_now, 0])
                data_counter = data_counter+1
            else:
                if dataRecolected:
                    agregar_datos([data_counter, 0, time_now, porcentaje])
                else:
                    agregar_datos([data_counter, 0, time_now, 0])
                data_counter = data_counter+1

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    
    no_detecting = True
    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_path
# ------------------------------------- . ------------------------------------ #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='penny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()


# --------------------------- Funcion de la Camara --------------------------- #
def cameraCV():
    global trigger, no_detecting
    cam = cv2.VideoCapture(0)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Conteo", frame)

        k = cv2.waitKey(1)
        
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32 and no_detecting:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            imOrig = Image.open(img_name)
            photoOrig = ImageTk.PhotoImage(imOrig)
            labelFotoOrig.configure(image=photoOrig)

            infFile = detect(img_name)
            print(infFile)

            im = Image.open(infFile)
            # print(file)
            # im = im.resize((500, 400))
            photo = ImageTk.PhotoImage(im)
            print(photo)
            labelFoto.configure(image=photo)

        elif trigger == True and no_detecting:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            imOrig = Image.open(img_name)
            photoOrig = ImageTk.PhotoImage(imOrig)
            labelFotoOrig.configure(image=photoOrig)

            infFile = detect(img_name)
            print(infFile)

            im = Image.open(infFile)
            # print(file)
            # im = im.resize((500, 400))
            photo = ImageTk.PhotoImage(im)
            print(photo)
            labelFoto.configure(image=photo)
            
            trigger = False

    cam.release()

    cv2.destroyAllWindows()
# ------------------------------------- . ------------------------------------ #

# ---------------------------- Funcion del Modbus ---------------------------- #
def modbus():
    global color
    global trigger
    pastDI0 = False
    di0 = False
    while True:
        try:
            print("Conectando modbus...")
            c = ModbusClient(host="10.0.0.1", port=502, unit_id=1, auto_open=True)

            while True:
                DIs = c.read_coils(0, 8)
                if DIs:
                    di0 = DIs[0]
                    if pastDI0 != di0:
                        if di0:
                            print("Se encendió 0")
                            trigger = True
                        pastDI0 = di0
                else:
                    print("read error")
                # print(c.is_open)
                #try:
                if color == 1:
                    verde=True
                    amarillo=False
                    rojo = False
                if color == 2:
                    verde = False
                    amarillo = True
                    rojo = False
                if color == 3:
                    verde = False
                    amarillo = False
                    rojo = True
                if color == 4:
                    verde = True
                    amarillo = True
                    rojo = True
                c.write_single_coil(16,verde)
                c.write_single_coil(17,amarillo)
                c.write_single_coil(18,rojo)
                    #c.write_multiple_registers(0,color)
                    #pass
                #except:
                #    print("Error al escribir")
                if c.is_open == False:
                    print("Modulo modbus desconectado")
                    break
                time.sleep(.1)
        except:
            print("No se puede conectar el modulo")
            time.sleep(5)
# ------------------------------------- . ------------------------------------ #

# -------------- Variables y funcion para calcular la eficiencia ------------- #
global perList
perList = []

def calcPer(piezas, tot, szProm):
    global color, perTot, perList, dataRecolected
    per = piezas / tot
    #print(per)
    perList.append(per)
    #print(perList)

    minYell = 0.97
    minGreen = 0.985

    if(len(perList)>=szProm):
        dataRecolected = True
        perTot = sum(perList)/szProm
        perList.pop(0)
        print("Promedio: " + str(perTot) )


        if perTot < minYell:
            print("Activa rojo")
            color = 3
        elif perTot > minGreen:
            print("Activa Verde")
            color = 1
        else:
            print("activa amarillo")
            color = 2
# ------------------------------------- . ------------------------------------ #

# ----------------------- Procesos y bucles principales ----------------------- #
threads = list()
tserv = threading.Thread(target=cameraCV)
threads.append(tserv)
tserv.start()

threads = list()
tserv = threading.Thread(target=modbus)
threads.append(tserv)
tserv.start()

root.config(menu=menubar)
root.mainloop()
# ------------------------------------- . ------------------------------------ #