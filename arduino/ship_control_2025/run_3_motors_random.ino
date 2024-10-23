void run_3_motors_random()
{
  run_3_motors();
  if(steeringMotor.distanceToGo() == 0 && throttleMotor.distanceToGo() == 0 && gearMotor.distanceToGo() == 0)
  {
    steeringValue = random(-STEERING_MAX_ANGLE, STEERING_MAX_ANGLE);
    throttleValue = random(1, 100);
    gearValue = random(1, GEAR_POSITIONS);
    print_value();
    delay(2000);
  }
}