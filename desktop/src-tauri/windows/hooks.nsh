!macro NSIS_HOOK_POSTINSTALL
  IfFileExists "$LOCALAPPDATA\Programs\TraceMemo\TraceMemo.exe" tracememo_done
  IfFileExists "$LOCALAPPDATA\Programs\tracememo\TraceMemo.exe" tracememo_done
  StrCpy $1 0
  tracememo_registry_loop:
    EnumRegKey $2 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall" $1
    StrCmp $2 "" tracememo_install
    ReadRegStr $3 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$2" "DisplayName"
    StrCmp $3 "TraceMemo" tracememo_done
    IntOp $1 $1 + 1
    Goto tracememo_registry_loop
  tracememo_install:
  DetailPrint "Installing bundled TraceMemo for local WeChat access..."
  ExecWait '"$INSTDIR\resources\tracememo\TraceMemo-setup.exe" /S' $0
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "TraceMemo installation returned code $0. Knowledge Garden is installed, but the WeChat connector may need repair."
  ${EndIf}
  tracememo_done:
!macroend
