void emergency_value()
{
   // Hiển thị trạng thái nút nhấn lên Serial Monitor
  Serial.print("Emergency (D2): ");
  Serial.println(digitalRead(emergency_buttonPin));
  digitalWrite(buzz, HIGH);
  // if (digitalRead(buttonPin) == 1) {
  //   Serial.println("1");
  // } 
  // else {
  //   Serial.println("0");
  // }
}