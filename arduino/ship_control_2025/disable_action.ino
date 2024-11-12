void disable_action()
{
  detachInterrupt(digitalPinToInterrupt(DISABLE_PIN));
  if (sta_disable == 0 && digitalRead(DISABLE_PIN) == LOW){
    delay(30);
    while(digitalRead(DISABLE_PIN) == 1){
      sta_disable = 1;
      Serial.println("disable");
      digitalWrite(BUZZER_PIN, LOW);
    }

  }
  else if (sta_disable == 1 && digitalRead(DISABLE_PIN) == LOW){
    delay(30);
    while(digitalRead(DISABLE_PIN) == 1){
      sta_disable = 0;
      Serial.println("enable");
      digitalWrite(BUZZER_PIN, HIGH);
    }
  }
  attachInterrupt(digitalPinToInterrupt(DISABLE_PIN), disable_action, RISING);
}