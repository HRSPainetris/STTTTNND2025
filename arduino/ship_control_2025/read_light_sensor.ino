void read_light_sensor()
{
  light_value = analogRead(LIGHT_SENSOR_PIN);
  if (light_value < light_threshold) {
    sta_light = 0;
    Serial.println("night");
  }
  else {
    sta_light = 1;
    Serial.println("day");
  }
}