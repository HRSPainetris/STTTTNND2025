void read_light_sensor()
{
  light_value = analogRead(LIGHT_SENSOR_PIN);
  // Serial.print("L: "); Serial.println(light_value);
  if (light_value < light_threshold) {
    sta_light = 0;
    Serial.println("day");
  }
  else {
    sta_light = 1;
    Serial.println("night");
  }
}