void start_button_value()
{
   // Hiển thị trạng thái nút nhấn lên Serial Monitor
  Serial.print("Start value (D3): ");
  Serial.println(digitalRead(start_buttonPin));
} 