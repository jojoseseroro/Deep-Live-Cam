; NSIS template for DeepLiveCam installer (template.nsi)
; Customize this file to add model files, shortcuts, and install options.

!define APP_NAME "DeepLiveCam-Standalone"
!define APP_VERSION "0.0.0"
!define DESCRIPTION "DeepLiveCam Standalone (Windows)"
!define INSTALL_DIR "$PROGRAMFILES\\${APP_NAME}"

OutFile "${APP_NAME}-Setup-${APP_VERSION}.exe"
InstallDir ${INSTALL_DIR}

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "dist\\*"
  ; Optionally download non-redistributable models here
  ; TODO: Add model download and verification UI via installer plugin or external script
  CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\DeepLiveCam.exe"
SectionEnd

Section "Uninstall"
  Delete "$INSTDIR\\DeepLiveCam.exe"
  RMDir /r "$INSTDIR"
  Delete "$DESKTOP\\${APP_NAME}.lnk"
SectionEnd
