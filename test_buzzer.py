from gpiozero import Buzzer
import time

buzzer = Buzzer(17)

print("Starting buzzer test...")

try:
    buzzer.off()
    time.sleep(2)

    while True:
        buzzer.on()
        print("Buzzer ON")
        time.sleep(0.3)

        buzzer.off()
        print("Buzzer OFF")
        time.sleep(1)

except KeyboardInterrupt:
    buzzer.off()
    print("Stopping buzzer test...")
