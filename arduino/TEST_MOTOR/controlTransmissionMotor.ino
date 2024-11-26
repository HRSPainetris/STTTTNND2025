void controlTransmissionMotor() {

  digitalWrite(TRANSMISSION_DIR_PIN, HIGH); 
  for (int i = 0; i < n_steps; i++) {
    digitalWrite(TRANSMISSION_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(TRANSMISSION_STEP_PIN, LOW);
    delayMicroseconds(500);
  }
  digitalWrite(TRANSMISSION_ENA_PIN, HIGH);
  Serial.println("DoneG");
}
