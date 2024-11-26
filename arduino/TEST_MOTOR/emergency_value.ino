void emergency_value()
{
   // Hiển thị trạng thái nút nhấn lên Serial Monitor
  Serial.print("Emergency (D2): ");
  Serial.println(digitalRead(emergency_buttonPin));
  while (digitalRead(emergency_buttonPin)==1)
  {
  digitalWrite(STEERING_ENA_PIN, HIGH);
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  digitalWrite(TRANSMISSION_ENA_PIN, HIGH);
  }
}