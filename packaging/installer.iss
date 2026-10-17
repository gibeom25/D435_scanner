; D435 Scanner 설치 프로그램 (Inno Setup 6)
; 먼저 packaging\build.bat 로 dist\D435 Scanner 를 만든 뒤 컴파일한다 (build.bat이 둘 다 한다).

#define AppName "D435 Scanner"
#define AppVersion "1.2.1"
#define AppExe "D435 Scanner.exe"

[Setup]
AppId={{60F1AB26-22B1-49FF-8BD1-C1B6A6E3ADDE}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=gibeom
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; 관리자 권한 없이 사용자 폴더에 설치 (원하면 설치 화면에서 '모든 사용자'로 바꿀 수 있음)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\installer
OutputBaseFilename=D435_Scanner_Setup_{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕 화면에 바로가기 만들기"; GroupDescription: "바로가기:"

[Files]
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{#AppName} 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 로그만 지운다. 스캔 데이터(저장 폴더)는 제거해도 남겨 둔다.
Type: filesandordirs; Name: "{app}\logs"

[Messages]
FinishedLabel=설치가 끝났습니다.%n%nRealSense D435를 USB 3 포트(파란색)에 연결하고 [{#AppName}]을 실행하세요. 별도 드라이버는 필요 없습니다.

[Code]
var
  DataPage: TInputDirWizardPage;

function ConfigPath: String;
begin
  Result := ExpandConstant('{userappdata}\D435Scanner\config.json');
end;

procedure InitializeWizard;
begin
  DataPage := CreateInputDirPage(wpSelectDir, '스캔 저장 폴더',
    '스캔 결과(녹화·메시)를 어디에 저장할까요?',
    '스캔 하나에 수백 MB가 쓰일 수 있습니다. 여유 공간이 넉넉한 드라이브를 고르세요. ' +
    '나중에 프로그램의 [폴더 바꾸기]로 언제든 바꿀 수 있습니다.',
    False, '');
  DataPage.Add('');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  { 이미 설정한 적이 있으면(재설치·업데이트) 묻지 않고 그대로 둔다 }
  Result := (PageID = DataPage.ID) and FileExists(ConfigPath);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = DataPage.ID) and (DataPage.Values[0] = '') then
  begin
    { 사용자 폴더에 설치했으면 설치 폴더 안, Program Files면 문서 폴더 }
    if IsAdminInstallMode then
      DataPage.Values[0] := ExpandConstant('{userdocs}\D435 Scanner\scans')
    else
      DataPage.Values[0] := ExpandConstant('{app}\scans');
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Dir, Json: String;
  Lines: TArrayOfString;
begin
  if (CurStep = ssPostInstall) and not FileExists(ConfigPath) and (DataPage.Values[0] <> '') then
  begin
    Dir := DataPage.Values[0];
    ForceDirectories(Dir);
    ForceDirectories(ExtractFileDir(ConfigPath));
    Json := Dir;
    StringChangeEx(Json, '\', '\\', True);
    SetArrayLength(Lines, 1);
    Lines[0] := '{"scans_dir": "' + Json + '"}';
    SaveStringsToUTF8File(ConfigPath, Lines, False);
  end;
end;
