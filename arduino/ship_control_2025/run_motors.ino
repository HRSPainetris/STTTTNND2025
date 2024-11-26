void run_motors(){
  if(digitalRead(E_STOP_PIN) == 0){ 
    steeringMotor.run();
    throttleMotor.run();
    gearMotor.run();
      // All motors ve trang thai nghi khi hoan thanh
    unsigned long currentTime = millis();
    if (steeringMotor.distanceToGo() == 0 && throttleMotor.distanceToGo() == 0 && gearMotor.distanceToGo() == 0 && readytorun && currentTime - lasttime >= timeoutPeriod){
      Serial.println("done");
      readytorun = 0;
    }
  }
}
