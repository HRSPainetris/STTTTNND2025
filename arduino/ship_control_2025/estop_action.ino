void estop_action()
{
  Serial.println("estop");
  while (digitalRead(E_STOP_PIN)==1)
  {
  sta_estop = 1;
  Serial.print("E-STOP pressed: "); Serial.println(digitalRead(E_STOP_PIN));
  // Release all motor
  release_motors();
  }
  sta_estop = 0;
}