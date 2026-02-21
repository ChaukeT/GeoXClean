; =============================================================================
; GeoX - Geological Block Model Viewer
; Inno Setup Installer Script
; =============================================================================
;
; Build with Inno Setup 6.x:
;   1. Install Inno Setup from https://jrsoftware.org/isinfo.php
;   2. Open this file in Inno Setup Compiler
;   3. Click Build -> Compile (or press Ctrl+F9)
;
; Prerequisites:
;   - Run PyInstaller first: pyinstaller GeoX.spec --noconfirm --clean
;   - Ensure dist/GeoX/GeoX.exe exists
;
; =============================================================================

#define AppName "GeoX"
#define AppFullName "GeoX - Geological Block Model Viewer"
#define AppVersion "1.1.0"
#define AppPublisher "GeoX Development Team"
#define AppURL "https://github.com/geox"
#define AppExeName "GeoX.exe"
#define AppCopyright "Copyright (C) 2024-2026 GeoX Development Team"

; Source directory (relative to this .iss file)
#define SourceDir "..\dist\GeoX"

[Setup]
; Unique application ID - DO NOT CHANGE after first release
AppId={{8E4F2A17-3F22-4F33-86B9-0C74E7B8B5A2}}
AppName={#AppFullName}
AppVersion={#AppVersion}
AppVerName={#AppFullName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
AppCopyright={#AppCopyright}

; Installation directories
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes

; Output settings
OutputDir=..\dist
OutputBaseFilename=GeoX_Setup_{#AppVersion}
SetupIconFile=..\block_model_viewer\assets\branding\geox_icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

; Compression (LZMA2 is best for large apps)
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; Installer appearance
WizardStyle=modern
WizardSizePercent=120
WizardImageFile=compiler:WizModernImage.bmp
WizardSmallImageFile=compiler:WizModernSmallImage.bmp

; Privileges (lowest = user install, admin = all users)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; 64-bit support
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Application management
CloseApplications=yes
CloseApplicationsFilter={#AppExeName}
RestartApplications=yes
MinVersion=10.0

; Uninstall settings
UninstallDisplayName={#AppFullName}
Uninstallable=yes
CreateUninstallRegKey=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "Create a &Quick Launch icon"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable
Source: "{#SourceDir}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Internal dependencies (_internal folder from PyInstaller)
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Optional: README and documentation
Source: "..\README.md"; DestDir: "{app}"; DestName: "README.txt"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion skipifsourcedoesntexist

; Optional: Sample data
Source: "..\sample_data\*"; DestDir: "{app}\sample_data"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Dirs]
; Create writable directories for user data
Name: "{localappdata}\{#AppName}"; Flags: uninsneveruninstall
Name: "{localappdata}\{#AppName}\logs"; Flags: uninsneveruninstall

[Icons]
; Start Menu
Name: "{group}\{#AppFullName}"; Filename: "{app}\{#AppExeName}"; Comment: "Launch {#AppFullName}"
Name: "{group}\{cm:UninstallProgram,{#AppFullName}}"; Filename: "{uninstallexe}"

; Desktop icon (if task selected)
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "Launch {#AppFullName}"

[Registry]
; File associations (optional - uncomment to enable)
; Root: HKCU; Subkey: "Software\Classes\.bmv"; ValueType: string; ValueData: "GeoX.BlockModel"; Flags: uninsdeletekey
; Root: HKCU; Subkey: "Software\Classes\GeoX.BlockModel"; ValueType: string; ValueData: "Block Model File"; Flags: uninsdeletekey
; Root: HKCU; Subkey: "Software\Classes\GeoX.BlockModel\DefaultIcon"; ValueType: string; ValueData: "{app}\{#AppExeName},0"; Flags: uninsdeletekey
; Root: HKCU; Subkey: "Software\Classes\GeoX.BlockModel\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExeName}"" ""%1"""; Flags: uninsdeletekey

; App settings path
Root: HKCU; Subkey: "Software\{#AppPublisher}\{#AppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#AppPublisher}\{#AppName}"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"; Flags: uninsdeletekey

[Run]
; Launch application after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppFullName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up log files on uninstall (user data preserved)
Type: filesandordirs; Name: "{localappdata}\{#AppName}\logs"

[Code]
// Custom Pascal script for advanced installer logic

function InitializeSetup(): Boolean;
begin
  Result := True;
  // Add any pre-installation checks here
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Post-installation tasks
    Log('GeoX installation completed successfully.');
  end;
end;
