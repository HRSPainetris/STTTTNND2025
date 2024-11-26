// Khai báo chân sử dụng
const int analogPin = A5;
const int emergency_buttonPin = 2;
const int start_buttonPin = 3;
const int buzz = 7;

void emergency_value();
void emergency_valuer();
void start_button_value();

void setup() {
  // Khởi tạo serial communication
  Serial.begin(9600);

  // Khai báo nút nhấn là đầu vào
  pinMode(emergency_buttonPin, INPUT);
  pinMode(start_buttonPin, INPUT_PULLUP);
  pinMode(buzz, OUTPUT);
  attachInterrupt(0, emergency_value, RISING); // gọi hàm in_gia_tri liên tục khi còn nhấn nút
  // attachInterrupt(0, emergency_valuer, LOW);
  attachInterrupt(1, start_button_value, FALLING); // gọi hàm in_gia_tri liên tục khi còn nhấn nút
}

void loop() {
  // Đọc giá trị analog từ chân A5
  int analogValue = analogRead(analogPin);
  // Hiển thị giá trị analog lên Serial Monitor
  Serial.print("Analog Value (A5): ");
  Serial.println(analogValue);
  // Đợi 500ms trước khi lặp lại
  delay(500);
  if (digitalRead(emergency_buttonPin)==0){
    // digitalWrite(buzz, LOW);
    emergency_valuer();
  } 
  // start_button_value();
  // emergency_value();
  // delay(100);
}