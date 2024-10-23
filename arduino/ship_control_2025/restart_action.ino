void restart_action()
{
  Serial.println("restart");
  // Chống nhiễu nút nhấn
  if (!digitalRead(RESTART_PIN) == 1) delay(100);
  if (!digitalRead(RESTART_PIN) == 1)
  {
    sta_restart = 1;
    wait_jetson1 = 1;
    // Release motor
    release_motors();

    // Cho jeston khoi tao lai chuong trinh va gui "new_record_created"
    while (wait_jetson1 == 1)
    {
      if (Serial.available())
      {
        inputString = Serial.readString();
        inputString.toLowerCase();
        //            Serial.print("Input string:   ");
        //            Serial.println(inputString);
        if (inputString.startsWith("new_record_created"))
        {
          //*** Hoan tat chu trinh binh thuong
          sta_restart = 0;
          wait_jetson1 = 0;
          break;
        }
      }
    }
  }

}