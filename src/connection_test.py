##########DEPENDENCIES#############

from dronekit import connect, VehicleMode,LocationGlobalRelative,APIException
import time
import socket
# import exceptions
import math
import argparse


#########FUNCTIONS#################

connection_string='/dev/ttyTHS1'
#baud=57600

def connectMyCopter():

	parser = argparse.ArgumentParser(description='commands')
	parser.add_argument('--connect')
	args = parser.parse_args()

	# connection_string = args.connect
	connection_string='/dev/ttyTHS1'


	vehicle = connect(connection_string,baud=57600,wait_ready=False)

	return vehicle

def arm():
	while vehicle.is_armable!=True:
		print("Waiting for vehicle to become armable.")
		time.sleep(1)
	print("Vehicle is now armable")

	vehicle.mode = VehicleMode("GUIDED")

	while vehicle.mode!='GUIDED':
		print("Waiting for drone to enter GUIDED flight mode")
		time.sleep(1)
	print("Vehicle now in GUIDED MODE. Have fun!!")

	vehicle.armed = True
	while vehicle.armed==False:
		print("Waiting for vehicle to become armed.")
		time.sleep(1)
	print("Vehicle is now armed.")

	return None

##########MAIN EXECUTABLE###########


vehicle = connectMyCopter()

vehicle.wait_ready('autopilot_version')
print('Autopilot version: %s'%vehicle.version)
while True:
	# print("Groundspeed: %s" % vehicle.groundspeed)
	# print ("Attitude: %s" % vehicle.attitude)
	# print("Velocity: %s" % vehicle.velocity)
	print("Groundspeed: {}. Velocity: {}.".format(vehicle.groundspeed,vehicle.velocity))

# arm()
