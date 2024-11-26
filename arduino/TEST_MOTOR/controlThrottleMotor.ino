void controlThrottleMotor() {

  digitalWrite(THROTTLE_DIR_PIN, HIGH); 
  for (int i = 0; i < n_steps; i++) {
    digitalWrite(THROTTLE_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(THROTTLE_STEP_PIN, LOW);
    delayMicroseconds(500);
  }
  digitalWrite(THROTTLE_ENA_PIN, HIGH);
  Serial.println("DoneT");
}
