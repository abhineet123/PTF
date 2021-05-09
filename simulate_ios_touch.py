from zxtouch.client import zxtouch
# device = zxtouch("192.168.1.70")
from zxtouch.touchtypes import *

import time

button_x = 1500
button_y = 1024

device = zxtouch("127.0.0.1")

while True:
    try:
        device.touch(TOUCH_DOWN, 5, button_x, button_y)
        # time.sleep(0.001)
        device.touch(TOUCH_UP, 5, button_x, button_y)
    except KeyboardInterrupt:
        break

device.disconnect()
