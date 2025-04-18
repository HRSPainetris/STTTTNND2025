void run_3_motors()
{

  // 1. Steering motor: Dong co xoay vo lang
  currentSteering = constrain(steeringValue, -STEERING_MAX_ANGLE, STEERING_MAX_ANGLE);
  digitalWrite(STEERING_ENA_PIN, LOW);
  // Tinh toan buoc quay cua banh lai
  steeringSteps = map(currentSteering, -STEERING_MAX_ANGLE * 2, STEERING_MAX_ANGLE * 2, 
                          -STEERING_STEPS_PER_REVOLUTION / 2, STEERING_STEPS_PER_REVOLUTION / 2);
  steeringSteps *= GEAR_RATIO;

  // 2. Throttle motor: Dong co keo can ga
  currentThrottle = constrain(throttleValue, 0, 100); // Gas tu 0-100%
  digitalWrite(THROTTLE_ENA_PIN, LOW);
  // Tinh toan buoc gas dua tren phan tram gia tri dau vao
  maxThrottleSteps = (MAX_THROTTLE_ROTATION_DEGREES / 360.0) * THROTTLE_STEPS_PER_REVOLUTION;
  throttleSteps = (currentThrottle / 100.0) * maxThrottleSteps;

  // 3. Gear motor: Dong co keo can so
  currentGear = constrain(gearValue, 1, GEAR_POSITIONS);
  digitalWrite(GEAR_ENA_PIN, LOW);
  gearSteps = (currentGear - 1) * GEAR_STEP; 

  // Chuyen dong 3 motors den vi tri muc tieu
  steeringMotor.moveTo(steeringSteps);
  throttleMotor.moveTo(throttleSteps);
  gearMotor.moveTo(gearSteps);

}