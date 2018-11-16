@echo on
netsh interface set interface "Wi-Fi" disabled
timeout /t 10
netsh interface set interface "Wi-Fi" enabled