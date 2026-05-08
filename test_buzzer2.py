from gpiozero import OutputDevice
import time

# try active_low because many buzzer modules work this way
buzzer = OutputDevice(17, active_high=False, initial_value=False)

print("Starting buzzer test...")

try:
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
