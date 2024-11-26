void release_motors()
{
  digitalWrite(STEERING_ENA_PIN, HIGH);  
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  digitalWrite(GEAR_ENA_PIN, HIGH);
}