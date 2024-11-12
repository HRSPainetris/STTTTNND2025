import serial
import time

###################################################
##             INTERFACE WITH ARDUINO            ##
###################################################
## Ket noi voi arduino 
try:                                                                                  
    arduino_module = serial.Serial(port = 'COM3', baudrate = 9600, timeout = 0.05)                         
    arduino_module.flush()
    print("Arduino connected successfully!")                                            
except:                                                                               
    print("Please check the port again") 

## Send message to Arduino
def send_message_to_arduino(arduino_module, message):
    # straight
    # left_50
    # right_50
    print("Message: ", message)
    arduino_module.write(message.encode())
    print("-Sent data to Arduino-")
    time.sleep(0.05)

## Read message from Arduino
def read_message_from_arduino(arduino_module):
    # day
    # night
    # estop
    # enable
    # disable
    # new_record
    message = arduino_module.readline() # doc tin hieu ve
    message = message.decode("utf-8").rstrip('\r\n') 
    print("Data from Arduino: ", message)
    return message

if __name__ == "__main__":
    try:
        while True:
            # Read input from the keyboard
            user_input = input("Enter message to send to Arduino (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            # Send the message to the Arduino
            send_message_to_arduino(arduino_module, user_input)
            
            if arduino_module.in_waiting > 0:
                # Read the message from the Arduino
                response = read_message_from_arduino(arduino_module)
                print("Response from Arduino: ", response)
    except KeyboardInterrupt:
        print("Exiting program")

    finally:
        arduino_module.close()
