void go_right()
{
  // Set gia tri goc xoay cho 3 motors
  steeringValue = read_angle;
  throttleValue = 50;
  gearValue = 2;
  // In gia tri vua set
  print_value();
  run_3_motors();
  lasttime = millis();
  readytorun = 1; 
}