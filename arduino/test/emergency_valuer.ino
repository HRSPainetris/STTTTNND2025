void emergency_valuer()
{
  Serial.print("Emergency (D2): ");
  Serial.println(digitalRead(emergency_buttonPin));
  digitalWrite(buzz, LOW);
}
