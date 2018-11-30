@echo on
start "PureVPN" "C:\Program Files (x86)\PureVPN\purevpn.exe"
timeout /t 120
start "Azureus" "C:\Program Files\Vuze\Azureus.exe"

:connected
REM wait for 5 minutes before next check
timeout /t 300
goto vpn_check

:vpn_check
netsh interface show interface name="Ethernet" ^
 |find "Connected">nul ^
   && goto connected ^
   || goto disconnected

:disconnected
REM restart
taskkill /f /im "Azureus.exe"
taskkill /f /im "purevpn.exe"
shutdown /r