@echo off
cd src
start "" cmd /k "mode con: cols=80 lines=15 && title Hai Trinh && conda activate khkt2025 && cd .. && python 0316_person_day.py"
start "" cmd /k "mode con: cols=80 lines=15 && title Drowsiness && conda activate khkt2025 && cd .. && python 0316_drowsiness.py"