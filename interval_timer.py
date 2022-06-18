# Import the required libraries
from tkinter import *
from pystray import MenuItem as item
import pystray
from PIL import Image, ImageTk

# Create an instance of tkinter frame or window
win = Tk()

win.title("Interval Timer")
# Set the size of the window
win.geometry("700x350")


# Define a function for quit the window
def quit_window(icon, item):
    icon.stop()
    win.destroy()


# Define a function to show the window again
def show_window(icon, item):
    icon.stop()
    win.after(0, win.deiconify())


# Hide the window and show on the system taskbar
def hide_window():
    win.withdraw()
    image = Image.open("favicon.ico")
    menu = (item('Quit', quit_window), item('Show', show_window))
    icon = pystray.Icon("name", image, "My System Tray Icon", menu)
    icon.run()


win.protocol('WM_DELETE_WINDOW', hide_window)

win.mainloop()
