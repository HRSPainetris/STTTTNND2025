void emergency_value()
{
  while (digitalRead(emergency_buttonPin)==1)
  {
  Serial.print("Emergency (D2): ");
  Serial.println(digitalRead(emergency_buttonPin));
  digitalWrite(STEERING_ENA_PIN, HIGH);  
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  digitalWrite(TRANSMISSION_ENA_PIN, HIGH);
  }
  // Serial.println("Continue");
}