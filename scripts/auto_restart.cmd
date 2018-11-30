@echo on
:loop
netsh interface set interface "Wi-Fi" enabled
netsh wlan connect name="UWS"
start "PureVPN" "C:\Program Files (x86)\PureVPN\purevpn.exe"
timeout /t 120
start "Azureus" "C:\Program Files\Vuze\Azureus.exe"
timeout /t 3600
taskkill /f /im "Azureus.exe"
taskkill /f /im "purevpn.exe"
netsh interface set interface "Wi-Fi" disabled
timeout /t 60
goto loop 