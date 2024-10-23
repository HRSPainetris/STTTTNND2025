void print_value()
{
  Serial.print("Steering: "); Serial.print(steeringValue);
  Serial.print("|| Throttle: "); Serial.print(throttleValue);
  Serial.print("|| Gear: "); Serial.println(gearValue);
}