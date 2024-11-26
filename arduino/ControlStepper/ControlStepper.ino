#include <AccelStepper.h>

#define STEERING_STEP_PIN 12
#define STEERING_DIR_PIN 13
#define STEERING_ENA_PIN 8

#define THROTTLE_STEP_PIN 4
#define THROTTLE_DIR_PIN 5
#define THROTTLE_ENA_PIN 9

#define TRANSMISSION_STEP_PIN 6
#define TRANSMISSION_DIR_PIN 7
#define TRANSMISSION_ENA_PIN 10
#define MOTOR_INTERFACE_TYPE 1

AccelStepper steeringMotor(MOTOR_INTERFACE_TYPE, STEERING_STEP_PIN, STEERING_DIR_PIN);
AccelStepper throttleMotor(MOTOR_INTERFACE_TYPE, THROTTLE_STEP_PIN, THROTTLE_DIR_PIN);
AccelStepper transmissionMotor(MOTOR_INTERFACE_TYPE, TRANSMISSION_STEP_PIN, TRANSMISSION_DIR_PIN);

const int STEERING_MAX_ANGLE = 180;
const int STEERING_STEPS_PER_REVOLUTION = 400; // thay doi theo driver
// const float GEAR_RATIO = 1.68; // Thay doi theo duong kinh pulley keo
const float GEAR_RATIO = 15; // Thay doi theo duong kinh pulley keo

const int THROTTLE_STEPS_PER_REVOLUTION = 200; // thay doi theo driver
const float MAX_THROTTLE_ROTATION_DEGREES = 90.0; 
const int TRANSMISSION_POSITIONS = 5; 
const int TRANSMISSION_STEP = 20;     

int currentSteering = 0;
int currentThrottle = 0;
int currentGear = 1;
int steeringValue = 0;
int throttleValue = 0;
int gearValue = 1;

const int emergency_buttonPin = 2;
void emergency_value();
void emergency_valuer();

void setup() {

  Serial.begin(115200);
  // Set max speed va gia toc cho tung dong co
  steeringMotor.setMaxSpeed(1000);
  steeringMotor.setAcceleration(500);
  throttleMotor.setMaxSpeed(1000);
  throttleMotor.setAcceleration(500);
  transmissionMotor.setMaxSpeed(1000);
  transmissionMotor.setAcceleration(500);

  steeringMotor.setCurrentPosition(0);
  throttleMotor.setCurrentPosition(0);
  transmissionMotor.setCurrentPosition(0);

  steeringMotor.setEnablePin(STEERING_ENA_PIN);
  throttleMotor.setEnablePin(THROTTLE_ENA_PIN);
  transmissionMotor.setEnablePin(TRANSMISSION_ENA_PIN);

  pinMode(emergency_buttonPin, INPUT);
  attachInterrupt(0, emergency_value, RISING);

  // all motor o trang thai nghi // 
  digitalWrite(STEERING_ENA_PIN, HIGH);
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  digitalWrite(TRANSMISSION_ENA_PIN, HIGH);

  Serial.print("S: ");
  Serial.print(steeringValue);
  Serial.print("| T: ");
  Serial.print(throttleValue);
  Serial.print("| G: ");
  Serial.println(gearValue);
}

void loop() {
    if (digitalRead(emergency_buttonPin)==0){
      // digitalWrite(buzz, LOW);
      emergency_valuer();
    }

    currentSteering = constrain(steeringValue, -STEERING_MAX_ANGLE, STEERING_MAX_ANGLE);
    digitalWrite(STEERING_ENA_PIN, LOW);
        // }
      // if (parsed >= 2) {
    currentThrottle = constrain(throttleValue, 0, 100); // Gas tu 0-100%
    digitalWrite(THROTTLE_ENA_PIN, LOW);
      // }

      // if (parsed == 3) {
    currentGear = constrain(gearValue, 1, TRANSMISSION_POSITIONS);
    digitalWrite(TRANSMISSION_ENA_PIN, LOW);
      // }

      // Tinh toan buoc quay cua banh lai
      int steeringSteps = map(currentSteering, -STEERING_MAX_ANGLE * 2, STEERING_MAX_ANGLE * 2, 
                              -STEERING_STEPS_PER_REVOLUTION / 2, STEERING_STEPS_PER_REVOLUTION / 2);
      steeringSteps *= GEAR_RATIO;

      // Tinh toan buoc gas dua tren phan tram gia tri dau vao
      int maxThrottleSteps = (MAX_THROTTLE_ROTATION_DEGREES / 360.0) * THROTTLE_STEPS_PER_REVOLUTION;
      int throttleSteps = (currentThrottle / 100.0) * maxThrottleSteps;

      // Chuyen dong motor den vi tri muc tieu
      steeringMotor.moveTo(steeringSteps);
      throttleMotor.moveTo(throttleSteps);
      transmissionMotor.moveTo((currentGear - 1) * TRANSMISSION_STEP);

  // Chay all motors
  steeringMotor.run();
  throttleMotor.run();
  transmissionMotor.run();
  // All motors ve trang thai nghi 
  if (steeringMotor.distanceToGo() == 0){ 
    digitalWrite(STEERING_ENA_PIN, HIGH);
    Serial.println("DisableSteering");
    }
  if (throttleMotor.distanceToGo() == 0) digitalWrite(THROTTLE_ENA_PIN, HIGH);
  if (transmissionMotor.distanceToGo() == 0) digitalWrite(TRANSMISSION_ENA_PIN, HIGH);
  if( (steeringMotor.distanceToGo() == 0) && (throttleMotor.distanceToGo() == 0) && (transmissionMotor.distanceToGo() == 0)){
      steeringValue = random(-180,180);
      throttleValue = random(1,100);
      gearValue = random(1,5);
      Serial.print("S: ");
      Serial.print(steeringValue);
      Serial.print("| T: ");
      Serial.print(throttleValue);
      Serial.print("| G: ");
      Serial.println(gearValue);
      delay(2000);
  }
}
