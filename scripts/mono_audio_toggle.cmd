@echo off

REM query audio state. 0x0: Stereo, 0x1: Mono

for /f "tokens=2*" %%a in ('reg query HKCU\Software\Microsoft\Multimedia\Audio /v AccessibilityMonoMixState') do set "var=%%b"

REM stop audio service

net stop audiosrv

REM Toggle mono/stereo setting in registry

if "%var%"=="0x1" (
  reg Add HKCU\Software\Microsoft\Multimedia\Audio /f /t REG_DWORD /v AccessibilityMonoMixState /d 0
) else (
  reg Add HKCU\Software\Microsoft\Multimedia\Audio /f /t REG_DWORD /v AccessibilityMonoMixState /d 1
)

REM restart audio service

net start audiosrv