import tkinter as tk
from tkinter import Tk, PhotoImage, Label
from tkinter import filedialog
import sys #Para finalizar el programa llamando a la función exit
from tkinter import filedialog  as fd #Ventanas de dialogo
from tkinter import messagebox as mb
from tkinter import *
import cv2
from cv2 import bitwise_or           #Libreria OpenCV para el procesamiento de imagenes
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
class Aplication:
    def __init__(self):
        self.ventana1 = tk.Tk()
        self.ventana1.config(bg="#A8A8D3")
        self.ventana1.title("Proyecto segmentación de TRIPS")
        self.ventana1.geometry("800x500")
        self.menu()
        
        # Configura la imagen de fondo
        self.background_image = PhotoImage(file="fondo3.gif")
        self.background_label = Label(self.ventana1, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        #BIENVENIDA
        Label(text = "BIENVENIDO A SEGMI-TRIPS", fg="black", font = ("Verdana", 18), bg="#8DC06D").place(x=195, y=0)
        
        # Crear un label que pueda cambiar de posición en donde se le pueda poner una imagen
        self.image_label = Label(self.ventana1, bg="white")
        self.image_label.place(x=250, y=100)

        #B O T O N E S
        self.botones()

        self.ventana1.mainloop()

    def menu(self):
        menubar1 = tk.Menu(self.ventana1)
        self.ventana1.config(menu=menubar1)
        opc1 = tk.Menu(menubar1, tearoff=0)
        opc1.add_command(label="Salir", command=self.salir)
        opc1.add_command(label="Guardar Imagen", command=self.guardar)
        opc1.add_command(label="Seleccionar Imagen", command=self.seleccionar)
        menubar1.add_cascade(label="File", menu=opc1) 
        pulgon = tk.Menu(menubar1, tearoff=0)
        pulgon.add_command(label="Negativo", command=self.negativoImagen)
        pulgon.add_command(label="Filtro promedio", command=self.promedio)
        pulgon.add_command(label="Filtro Sobel", command=self.sobel)
        menubar1.add_cascade(label="Pulgon", menu=pulgon) 
        hormigas = tk.Menu(menubar1, tearoff=0)
        hormigas.add_command(label="Filtro Gausseano", command=self.apply_gaussian_filter)
        hormigas.add_command(label="Expansión Sustracción", command=self.guardar)
        hormigas.add_command(label="Binarización", command=self.create_interface)
        hormigas.add_command(label="Laplaciano negativo", command=self.apply_negative_laplacian_filter)
        menubar1.add_cascade(label="Hormigas", menu=hormigas) 
        babosa = tk.Menu(menubar1, tearoff=0)
        babosa.add_command(label="Salir", command=self.salir)
        babosa.add_command(label="Guardar Imagen", command=self.guardar)
        babosa.add_command(label="Seleccionar Imagen", command=self.seleccionar)
        menubar1.add_cascade(label="Babosa", menu=babosa) 
        mosca = tk.Menu(menubar1, tearoff=0)
        mosca.add_command(label="Salir", command=self.salir)
        mosca.add_command(label="Guardar Imagen", command=self.guardar)
        mosca.add_command(label="Seleccionar Imagen", command=self.seleccionar)
        menubar1.add_cascade(label="MoscaBlanca", menu=mosca) 

    def salir(self):
        sys.exit(0)

    def guardar(self):
        nomArchivo = fd.asksaveasfilename(initialdir= "C:/Users/USER/OneDrive/Escritorio/AImages", title= "Guardar como", defaultextension=".png")
        if nomArchivo!='':
            self.image.save(nomArchivo)
            mb.showinfo("Información", "La imagen ha sido guardada correctamente.")
    def seleccionar(self):
        self.nomArchivo = fd.askopenfilename(initialdir= "C:/Users/USER/OneDrive/Escritorio/AImages", title= "Seleccionar Archivo", filetypes= (("Image files", "*.png; *.jpg; *.gif"),("todos los archivos", "*.*")))
        if self.nomArchivo!='':
            self.image = Image.open(self.nomArchivo)
            self.image = self.image.resize((270, 250))#Normalizamos la imagen
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.configure(image=self.photo)
        
            # Habilitar botones
            self.gray_button.configure(state="normal")
            self.exp_button.configure(state="normal")
            self.max_button.configure(state="normal")
            self.kirsch_button.configure(state="normal")


    def botones(self):
        self.gray_button = Button(self.ventana1, text="Escala de grises", width=2, height=2, bg="#E5DBF7", fg="black",font=("Helvetica", 10, "bold"), command=self.gray_scale, state="disabled")
        self.gray_button.place(x= 110, y= 380, width=120)

        self.exp_button = Button(self.ventana1, text="Ecualización Expo", width=2, height=2, bg="#E5DBF7", fg="black",font=("Helvetica", 10, "bold"), command=self.hist_contrast, state="disabled")
        self.exp_button.place(x= 250, y= 380, width=120)

        self.max_button = Button(self.ventana1, text="Filtro Max", width=2, height=2, bg="#E5DBF7", fg="black",font=("Helvetica", 10, "bold"), command=self.max_filter, state="disabled")
        self.max_button.place(x= 390, y= 380, width=120)

        self.kirsch_button = Button(self.ventana1, text="Kirsch", width=2, height=2, bg="#E5DBF7", fg="black",font=("Helvetica", 10, "bold"), command=self.kirsch_operator, state="disabled")
        self.kirsch_button.place(x= 550, y= 380, width=120)
#-------------------------------SEGEMETANCIÓN DE TRIPS-------------------------------------------
    def gray_scale(self):
        # Convertir imagen a escala de grises y actualizar etiqueta
        self.image = self.image.convert("L")
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)
        self.image_array = np.array(self.image)
        #self.image.show()
        cv2.imshow('Escala de grises', self.image_array)

    def hist_contrast(self): #Se ecualiza
        # Mejora de contraste por histograma y actualizar etiqueta
        self.image = ImageOps.equalize(self.image)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)
        self.image_array = np.array(self.image)
        cv2.imshow('Ecualización de la imagen', self.image_array) 

    def max_filter(self):
        # Filtro máximo y actualizar etiqueta
        self.image = self.image.filter(ImageFilter.MaxFilter(3))
        
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo)
        self.image_array = np.array(self.image)
        cv2.imshow('Filtro Máximo', self.image_array)

    def kirsch_operator(self):
        self.image = self.image.convert("L")
        #Convertir la imagen a un arreglo de numpy
        self.image_array = np.array(self.image)

        #Crear las máscaras de kirsch
        self.masks =[    np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=float),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=float),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=float),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=float),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=float),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=float),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=float),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=float)
        ]

        #Crear las matrices las matrices resultado para cada grado de Kirsch
        self.results= []
        for i in range(8):
            self.results.append(np.zeros_like(self.image_array))

        #Aplicar el operador de Kirsch para cada grado

        for i in range(8):
            mask = self.masks[i]
            for r in range(1, self.image_array.shape[0]-1):
                for c in range(1, self.image_array.shape[1]-1):
                    sub_image = self.image_array[r-1:r+2, c-1:c+2]
                    result = np.sum(sub_image * mask)
                    self.results[i][r, c] = np.sqrt(result ** 2)
        
        #Unir las matrices resultado para obtener la imagen final
        self.final_image_array = np.max(self.results, axis=0)
        #self.final_image = self.image.fromarray(self.final_image_array.astype(np.uint8))
        #self.final_image = Image.fromarray(self.final_image_array.astype(np.uint8))

        #Mostrar las imagenes resultantes
        #self.final_image.show()
        cv2.imwrite(self.nomArchivo, self.final_image_array)
        cv2.imshow('Filtro kirsch', self.final_image_array)
        for i in range(8):
            #result_image = Image.fromarray(self.results[i].astype(np.uint8))
            #result_image.show()
            cv2.imshow('Matrices', self.results)
#---------------------------------------SEGMENTACIÓN PULGON----------------------------------------------
    def negativoImagen(self):
        self.image_array2 = np.array(self.image)
        cv2.imshow('Original', self.image_array2)
        self.inverted_image = np.invert(self.image)

        self.image_array = np.array(self.inverted_image)
        cv2.imshow('Ecualización de la imagen', self.image_array) 
        filenameInvert = self.nomArchivo+"_negativo.jpg"                
        cv2.imwrite(filenameInvert, self.image_array)
        #self.final_image = Image.fromarray(self.inverted_image.astype(np.uint8))
        #self.final_image.show()
        print("Aqui va el negativo de una imagen")
        

    #Función Filtro Promedio
    def promedio(self):
        #Convertir la imagen a un arreglo de numpy
        self.image_array = np.array(self.image)
        #Crea el kernel
        self.kernel3x3 = np.ones((3,3),np.float32)/9.0
        self.kernel5x5 = np.ones((5,5),np.float32)/25.0

        #Filtra la imagen utilizando el kernel anterior
        self.salida3 = cv2.filter2D(self.image_array,-1,self.kernel3x3)
        self.salida5 = cv2.filter2D(self.image_array,-1,self.kernel5x5)
        #Convertimos de un arreglo numpy a imagen
        self.final_image = Image.fromarray(self.image_array.astype(np.uint8))
        self.final_image.show()
        print("Aqui va el filtro promedio")
        
    #Funcion que realiza la deteccion de bordes de sobel
    def sobel(self): #TODO: create a formula to get an "automatic" way to apply displacement 
        self.imagenew = self.image.convert("L")
        #Convertir la imagen a un arreglo de numpy
        self.image_array = np.array(self.imagenew)
        self.x = cv2.Sobel(self.image_array,cv2.CV_16S,1,0)
        self.y = cv2.Sobel(self.image_array,cv2.CV_16S,0,1)
        self.absX = cv2.convertScaleAbs(self.x)
        self.absY = cv2.convertScaleAbs(self.y)
        self.sobel = cv2.addWeighted(self.absX,0.5,self.absY,0.5,0)
        cv2.imshow('Sobel', self.sobel)
        #self.sobel.show() #Muestra la imagen despues de la funcion Sobel
        print("aqui va sobel")
#---------------------------------------SEGMENTACIÓN HORMIGAS----------------------------------------------
    def apply_gaussian_filter(self):
        # Lee la imagen
        image = cv2.imread(self.nomArchivo)
        kernel_size = 13  # Tamaño del kernel del filtro gaussiano (impar)
        sigma = 0 
        # Verifica si la imagen se ha cargado correctamente
        if image is None:
            print('No se pudo cargar la imagen.')
            return

        # Aplica el filtro gaussiano
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        cv2.imshow("Filtro Gausseano", filtered_image)
        

    def apply_negative_laplacian_filter(self):
        # Lee la imagen en escala de grises
        image = cv2.imread(self.nomArchivo, cv2.IMREAD_GRAYSCALE)
        # Aplica el filtro Laplaciano
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # Obtiene el valor máximo absoluto de la imagen filtrada
        max_value = np.max(np.abs(laplacian))

        # Normaliza y crea el filtro Laplaciano negativo
        negative_laplacian = (255 / max_value) * np.abs(laplacian)
        negative_laplacian = negative_laplacian.astype(np.uint8)
        cv2.imshow("Filtro Laplaciano", negative_laplacian)

    def create_interface(self):
            self.root = tk.Tk()
            self.root.title("Binarización")
            self.threshold = 127
            self.image_path = ""
            # Barra deslizante para ajustar el umbral
            self.threshold_scale = tk.Scale(self.root, from_=1, to=255, length=400,
                                            orient=tk.HORIZONTAL, label="Threshold",
                                            command=self.update_threshold)
            self.threshold_scale.set(self.threshold)
            self.threshold_scale.pack()

            self.update_threshold(self.threshold)

    def update_threshold(self, value):
        self.threshold = int(value)
        #self.imagenew = self.image.convert("L")
        #Convertir la imagen a un arreglo de numpy
        #self.image_array = np.array(self.imagenew)
        # Leer la imagen en escala de grises
        self.image_array = np.array(self.image)
        self.img = cv2.imread(self.nomArchivo)
        self.image_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Binarizar la imagen usando el umbral
        ret, binary_image = cv2.threshold(self.image_gray, self.threshold, 255, cv2.THRESH_BINARY)

        # Mostrar la imagen binarizada
        cv2.imshow("Imagen Binarizada", binary_image)
        self.root.mainloop()
#---------------------------------------SEGMENTACIÓN BABOSAS----------------------------------------------

#---------------------------------------SEGMENTACIÓN MOSCABLANCA----------------------------------------------         

aplication = Aplication()