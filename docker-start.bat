@echo off
echo Starting Docker build and deployment...
docker-compose up --build -d
echo Deployment triggered! Access your app at http://localhost:5000
pause
