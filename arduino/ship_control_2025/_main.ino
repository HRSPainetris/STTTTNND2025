void _main ()
{
  // Step 1: Doc tin hieu cua light sensor va gui Serial S1 =  "day" or "night" cho jetson
  // Step 2: Cho Jetson gui tin hieu S2 =  "straight" or "left_agl" or "right_agl"
  // Step 3: Giai ma tin hieu
    // straight
    // left_agl : left_
    // right_agl : right_50
  // Step 4: Sau khi chay xong thi gui cho Jetson "done"
    // Quay lai buoc 1

  // Interrupt:
    // Khi nhan nut e-stop thi:
      // Dung het tat ca dong co, gui Serial "estop" cho Jetson
    // Khi nhan nut restart thi:
      // Gui Serial "restart" cho Jetson
      // Cho Jetson gui tin hieu "new_record_created"
    // Khi nhan nut disable thi:
      // Tat coi arduino
      // Gui Serial "disable" cho Jetson

    // Step 1: Doc tin hieu cua light sensor va gui Serial S1 =  "day" or "night" cho jetson
    read_light_sensor();
    // Step 2: Cho Jetson gui tin hieu S2 =  "straight" or "left_agl" or "right_agl"

    if (Serial.available()) 
    {
      inputString = Serial.readString();  // Đọc chuỗi từ Serial
      inputString.toLowerCase();
      if (inputString.startsWith("straight"))
        {
          go_straight();
        }
      else if (inputString.startsWith("left_"))
        {
          // left_50
          // -> read_angle = 50
          read_angle = inputString.substring(5, 7).toInt();  // Lấy 2 ký tự sau "left_" và chuyển đổi chuỗi sang số nguyên
          go_left();
        }
      else if (inputString.startsWith("right_"))
        {
          read_angle = inputString.substring(6, 8).toInt();  // Lấy 2 ký tự sau "right_" và chuyển đổi chuỗi sang số nguyên
          go_right();
        }
      // Cho jeston khoi tao lai chuong trinh va gui "new_record_created"
      if (wait_jetson1 == 1 && inputString.startsWith("new_record_created"))
        {
            //*** Hoan tat chu trinh binh thuong
          sta_restart = 0;
          wait_jetson1 = 0;
          Serial.println("completecyle");
        }
    }

    // delay(200);
}