void _main ()
{
  // Step 1: Doc tin hieu cua light sensor va gui Serial S1 =  "day" or "night" cho jetson
  // Step 2: Cho Jetson gui tin hieu S2 =  "straight" or "left_agl" or "right_agl"
  // Step 3: Giai ma tin hieu
    // straight
    // left_agl
    // right_agl
  // Step 4: Sau khi chay xong thi gui cho Jetson "done"

  // Interrupt:
    // Khi nhan nut e-stop thi:
      // Dung het tat ca dong co, gui Serial "estop" cho Jetson
    // Khi nhan nut restart thi:
      // Gui Serial "restart" cho Jetson
      // Cho Jetson gui tin hieu "new_record_created"
      // Quay lai buoc 1
}