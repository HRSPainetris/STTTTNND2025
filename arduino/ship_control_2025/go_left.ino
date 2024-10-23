void go_left()
{
    // Set gia tri goc xoay cho 3 motors
  steeringValue = - read_angle;
  throttleValue = 70;
  gearValue = 2;
  // In gia tri vua set
  print_value();
  // Run 3 dong co theo goc vua set
  run_3_motors();
  delay(1500);
}