@echo off
cd /d <REPO_PATH>

call "C:\Users\\<USER_NAME>\Anaconda3\Scripts\activate.bat" <ENV_NAME>

python main.py

pause