import time
import board
import adafruit_dht

dht_device = adafruit_dht.DHT11(board.D27)

print("Starting DHT11 test...")

try:
    while True:
        try:
            temperature_c = dht_device.temperature
            humidity = dht_device.humidity

            print(f"Temperature: {temperature_c} C")
            print(f"Humidity: {humidity} %")
            print("----------------------")

        except RuntimeError as e:
            print("Sensor reading error:", e)

        time.sleep(2)

except KeyboardInterrupt:
    print("Stopping test...")
    dht_device.exit()
