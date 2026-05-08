import time
import board
import busio
import adafruit_ccs811

i2c = busio.I2C(board.SCL, board.SDA)
ccs = adafruit_ccs811.CCS811(i2c)

print("Waiting for sensor to be ready...")

while not ccs.data_ready:
    time.sleep(1)

while True:
    print("eCO2 =", ccs.eco2, "ppm")
    print("TVOC =", ccs.tvoc, "ppb")
    print("----------------------")
    time.sleep(2)
