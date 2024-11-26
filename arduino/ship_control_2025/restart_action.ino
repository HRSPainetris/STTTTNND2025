void restart_action()
{
  Serial.println("restart");
  sta_restart = 1;
  wait_jetson1 = 1;
}