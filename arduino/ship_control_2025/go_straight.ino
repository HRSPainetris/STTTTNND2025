void go_straight()
{
  // Set gia tri goc xoay cho 3 motors
  steeringValue = 0;
  throttleValue = 70;
  gearValue = 3;
  // In gia tri vua set
  print_value();
  run_3_motors();
  lasttime = millis();
  readytorun = 1; 
}