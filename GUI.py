from statistics import mean
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import time


width = 500  # canvas width
height = 500 # canvas height
center = height//2
white = (255, 255, 255) # canvas back
start = time.time()
times = []
started = False
def save(times):
    # save image to hard drive
    filename = "user_input.jpg"
    output_image.convert("L")
    output_image.save(filename)
    arr = np.array(output_image)
    print(arr.shape)

    grey = arr[:,:,0]
    times = np.array(times)
    start = times[0]
    times = np.subtract(times, start)
    print(times)
    np.savetxt("rectangledata.csv", grey, delimiter=",")
    generate_trajectory(grey)


def paint(event):
    #x1, y1 = (event.x - 1), (event.y - 1)
    #x2, y2 = (event.x + 1), (event.y + 1)
    x1, y1 = (event.x), (event.y)
    canvas.create_rectangle(x1, y1, x1, y1, fill="black",width=0.5)
    draw.rectangle([x1, y1, x1, y1],fill="black",width=0.5)
    cur = time.time()-start
    times.append(cur)

def generate_trajectory(arr):
    #135,420
    test = np.array(np.where(arr==0))
    #subtract 250 from the x and ydirection:
    test[0] = np.subtract(250,test[0])
    test[1] = np.subtract(test[1],250)
    test = np.multiply(test, 8/250.0)
    test[[0,1]] = test[[1,0]]
    print("TIMES:-------------")
    print(len(times))
    print("TEST:-----------------")
    test = np.vstack([test, times])
    print(test)
    print(test.shape)
    np.savetxt("trajectory.csv", test, delimiter=",")

def sg_filter(arr, n_dem):
    """data = [];
    for dem = 1:n_demonstrations
        x_obs_dem = x_obs{dem}(1:2,:)';
        dt = mean(diff(x_obs{dem}(3,:)')); % Average sample time (Third dimension contains time)
        dx_nth = sgolay_time_derivatives(x_obs_dem, dt, 2, 3, 15);
        if (struct_output)
            data{dem} = [dx_nth(:,:,1),dx_nth(:,:,2)]';
        else
            data = [data [dx_nth(:,:,1),dx_nth(:,:,2)]'];
        end
    """
    for i in range(n_dem):
        coords = arr[n_dem][:]
        dt = mean()
app = Tk()
image = ImageTk.PhotoImage(Image.open("/Users/MonicaChan/Desktop/UROP/Python Implementation/rectangles.png"))

output_image = Image.new("RGB", (width, height), white)

canvas = Canvas(app, bg='black')
canvas.pack(anchor='nw', fill='both', expand=1)

canvas.create_image(0,0, image=image, anchor='nw')

app.geometry("500x500")

draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button=Button(text="save",command= lambda: save(times))
button.pack()

#button=Button(text="clear",command=save)
#button.pack()

app.mainloop()