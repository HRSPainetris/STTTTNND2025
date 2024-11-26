void emergency_valuer()
{
  Serial.print("Emergency (D2): ");
  Serial.println(digitalRead(emergency_buttonPin));
  digitalWrite(STEERING_ENA_PIN, LOW);
  digitalWrite(THROTTLE_ENA_PIN, LOW);
  digitalWrite(TRANSMISSION_ENA_PIN, LOW);
}
