// KHAI BAO THU VIEN
#include <AccelStepper.h>

//////////////////////////////////////////
// KHAI BAO CHUONG TRINH CON
void _main();

// 4. Nut nhan e-stop
void estop_action();
void release_motors();

// 5. Nut nhan restart
void restart_action();

// 6. Cam bien anh sang
void read_light_sensor();

// 7. In gia tri
void print_value();

// 8. Dieu khien tau
void go_straight(); // Di thang
void go_left(); // Di sang trai
void go_right(); // Di sang phai
void run_3_motors();
void run_3_motors_random();

//////////////////////////////////////////
// KHAI BAO CHAN
#define MOTOR_INTERFACE_TYPE 1 // Loai giao tiep dong co cua thu vien AccelStepper (1: DRIVER)
// 1. Steering motor: Dong co xoay vo lang
#define STEERING_ENA_PIN 8
#define STEERING_DIR_PIN 13
#define STEERING_STEP_PIN 12
AccelStepper steeringMotor(MOTOR_INTERFACE_TYPE, STEERING_STEP_PIN, STEERING_DIR_PIN);

// 2. Throttle motor: Dong co keo can ga
#define THROTTLE_ENA_PIN 9
#define THROTTLE_DIR_PIN 5
#define THROTTLE_STEP_PIN 4
AccelStepper throttleMotor(MOTOR_INTERFACE_TYPE, THROTTLE_STEP_PIN, THROTTLE_DIR_PIN);

// 3. Gear motor: Dong co keo can so
#define GEAR_ENA_PIN 10
#define GEAR_DIR_PIN 7
#define GEAR_STEP_PIN 6
AccelStepper gearMotor(MOTOR_INTERFACE_TYPE, GEAR_STEP_PIN, GEAR_DIR_PIN);

// 4. Nut nhan e-stop
#define E_STOP_PIN 2

// 5. Nut nhan restart
#define RESTART_PIN 3

// 6. Cam bien anh sang
const int LIGHT_SENSOR_PIN = A5;

//////////////////////////////////////////
// KHAI BAO BIEN
String inputString; // Du lieu doc duoc tu Serial
int read_angle = 0;

// 1. Steering motor: Dong co xoay vo lang
const int STEERING_MAX_ANGLE = 90; // Goc banh lai toi da cua thuyen
const int STEERING_STEPS_PER_REVOLUTION = 400; // So xung dieu khien khi dong co quay 1 vong (thay doi theo driver)
// const float GEAR_RATIO = 1.68; // Ti le goc quay cua tau vs so vong vo lang
const float GEAR_RATIO = 15; // Ti le goc quay cua tau vs so vong vo lang
int currentSteering = 0;
int steeringValue = 0;

// 2. Throttle motor: Dong co keo can ga
const int THROTTLE_STEPS_PER_REVOLUTION = 200; // So xung dieu khien khi dong co quay 1 vong (thay doi theo driver)
const float MAX_THROTTLE_ROTATION_DEGREES = 90.0; 
int currentThrottle = 0;
int throttleValue = 0;

// 3. Gear motor: Dong co keo can so
const int GEAR_POSITIONS = 3; // So luong cap so (1:LUI, 2:KHONG SO; 3: TIEN)
const int GEAR_STEP = 40;     
int currentGear = 1;
int gearValue = 1;

// 4. Nut nhan e-stop
int sta_estop = 0;

// 5. Nut nhan restart
int sta_restart = 0;
int wait_jetson1 = 0;

// 6. Cam bien anh sang
int light_value = 0; // Gia tri analog cua cam bien anh sang
int light_threshold = 400; // Gia tri nguong quy dinh ngay hoac dem
int sta_light = 1; // Luu gia tri dieu kien anh sang (0: Night; 1: Day)

void setup() {
  Serial.begin(115200);
  //////////////////////////////////////////
  // KHAI BAO MAX_SPEED | GIA TOC | ENABLE_PIN | RELEASE MOTOR | CURRENT_POSITION | 
  // 1. Steering motor: Dong co xoay vo lang
  steeringMotor.setMaxSpeed(1000);
  steeringMotor.setAcceleration(500);
  steeringMotor.setEnablePin(STEERING_ENA_PIN);
  digitalWrite(STEERING_ENA_PIN, HIGH);
  steeringMotor.setCurrentPosition(0);

  // 2. Throttle motor: Dong co keo can ga
  throttleMotor.setMaxSpeed(1000);
  throttleMotor.setAcceleration(500);
  throttleMotor.setEnablePin(THROTTLE_ENA_PIN);
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  throttleMotor.setCurrentPosition(0);

  // 3. Gear motor: Dong co keo can so
  gearMotor.setMaxSpeed(1000);
  gearMotor.setAcceleration(500);
  gearMotor.setEnablePin(GEAR_ENA_PIN);
  digitalWrite(GEAR_ENA_PIN, HIGH);
  gearMotor.setCurrentPosition(0);

  // 4. Nut nhan e-stop
  pinMode(E_STOP_PIN, INPUT);
  attachInterrupt(0, estop_action, RISING);

  // 5. Nut nhan restart
  pinMode(RESTART_PIN, INPUT_PULLUP);
  attachInterrupt(1, restart_action, FALLING);
}

void loop() {
  // run_3_motors_random();
  // check_code();
  _main();
}
