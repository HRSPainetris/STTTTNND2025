void disable_action()
{
  if (digitalRead(DISABLE_PIN) == 1) delay(100);
  if(digitalRead(DISABLE_PIN) == 1 && sta_disable == 0)
  {
    Serial.println("disable");
    digitalWrite(BUZZER_PIN, LOW);
    sta_disable = 1;
  }
  else if(!digitalRead(DISABLE_PIN) == 1 && sta_disable == 1)
  {
   Serial.println("enable");
   sta_disable = 0;
  }
}