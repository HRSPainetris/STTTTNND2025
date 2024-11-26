void controlSteeringMotor() {

  digitalWrite(STEERING_DIR_PIN, HIGH); 
  for (int i = 0; i < n_steps; i++) {
    digitalWrite(STEERING_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(STEERING_STEP_PIN, LOW);
    delayMicroseconds(500);
  }
  digitalWrite(STEERING_ENA_PIN, HIGH);
  Serial.println("DoneS");
}
