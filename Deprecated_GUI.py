import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageTk, ImageDraw
import PIL.Image
from tkinter import *

width = 500  # canvas width
height = 500 # canvas height
center = height//2
white = (255, 255, 255) # canvas back

def save():
    # save image to hard drive
    filename = "user_input.png"
    output_image.save(filename)
    print(np.array(output_image))

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

master = Tk()

# create a tkinter canvas to draw on
canvas = Canvas(master, width=width, height=height, bg='white')
canvas.pack()

# create an empty PIL image and draw object to draw on
"""
objs{1}.pos = [-100 -100 200 96]; yellow region
objs{2}.pos = [2 -8 4 14]; pink region
objs{3}.pos = [2 6 4 2]; green region
objs{2}.color = [.8 .5 .5]; % salmon
objs{3}.color = [.3 .8 .3]; % green
objs{1}.color = [1. .8 .3]; % yellow"""
output_image = PIL.Image.new("RGB", (width, height), white)
img1 = ImageDraw.Draw(output_image)  
img1.rectangle(xy = [(0, 350), (500,500)], fill = (255,204,77)) #yellow region
img1.rectangle(xy = [(500,500), (400,0)], fill = (204, 128,128)) #pink region
img1.rectangle(xy = [(500,0), (400,100)], fill = (77, 204, 77)) #green region

draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button=Button(text="save",command=save)
button.pack()

master.mainloop()

#vector experimentaion
"""x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))

u = x/np.sqrt(x**2 + y**2)
v = y/np.sqrt(x**2 + y**2)

plt.quiver(x,y,u,v)
plt.show()"""