@echo off
cd /d c:\Users\og\Desktop\CrewAi
echo Configuring git...
git config --global credential.helper wincred
echo Pushing commits...
git push -u origin main
echo Push complete!
pause
