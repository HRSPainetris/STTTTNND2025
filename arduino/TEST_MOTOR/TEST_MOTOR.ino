
#define STEERING_STEP_PIN 11
#define STEERING_DIR_PIN 12
#define STEERING_ENA_PIN 8

#define THROTTLE_STEP_PIN 4
#define THROTTLE_DIR_PIN 5
#define THROTTLE_ENA_PIN 9


#define TRANSMISSION_STEP_PIN 6
#define TRANSMISSION_DIR_PIN 7
#define TRANSMISSION_ENA_PIN 10

int n_steps = 10000;
const int emergency_buttonPin = 2;

void emergency_value();
void emergency_valuer();
void controlThrottleMotor();
void controlSteeringMotor();
void controlTransmissionMotor();

void setup() {
  Serial.begin(115200);


  pinMode(STEERING_STEP_PIN, OUTPUT);
  pinMode(STEERING_DIR_PIN, OUTPUT);
  pinMode(STEERING_ENA_PIN, OUTPUT);

  pinMode(THROTTLE_STEP_PIN, OUTPUT);
  pinMode(THROTTLE_DIR_PIN, OUTPUT);
  pinMode(THROTTLE_ENA_PIN, OUTPUT);

  pinMode(TRANSMISSION_STEP_PIN, OUTPUT);
  pinMode(TRANSMISSION_DIR_PIN, OUTPUT);
  pinMode(TRANSMISSION_ENA_PIN, OUTPUT);


  digitalWrite(STEERING_ENA_PIN, LOW); //khoa dong co
  digitalWrite(THROTTLE_ENA_PIN, LOW); //khoa dong co
  digitalWrite(TRANSMISSION_ENA_PIN, LOW); //khoa dong co

  pinMode(emergency_buttonPin, INPUT);
  attachInterrupt(0, emergency_value, RISING); // gọi hàm in_gia_tri liên tục khi còn nhấn nút
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');

    if (input.startsWith("S")) {
      controlSteeringMotor();
    }


    if (input.startsWith("T")) {
      controlThrottleMotor();
    }


    if (input.startsWith("G")) {
      controlTransmissionMotor();
    }
        
    if (input.startsWith("A")) {
      controlSteeringMotor();
      controlThrottleMotor();
      controlTransmissionMotor();
    }

    if (input.startsWith("R")) {
      digitalWrite(STEERING_ENA_PIN, HIGH);
      digitalWrite(THROTTLE_ENA_PIN, HIGH);
      digitalWrite(TRANSMISSION_ENA_PIN, HIGH);
    }
  }

  if (digitalRead(emergency_buttonPin)==0){
    // digitalWrite(buzz, LOW);
    emergency_valuer();
  }

}