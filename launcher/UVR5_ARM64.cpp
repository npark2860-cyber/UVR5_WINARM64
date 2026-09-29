#include <windows.h>
#include <string>
#include <vector>

static std::wstring GetModulePath() {
    std::vector<wchar_t> buffer(32768);
    DWORD len = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size()) {
        return L"";
    }
    return std::wstring(buffer.data(), len);
}

static std::wstring ParentDirectory(const std::wstring& path) {
    const size_t pos = path.find_last_of(L"\\/");
    if (pos == std::wstring::npos) {
        return L"";
    }
    return path.substr(0, pos);
}

static std::wstring Quote(const std::wstring& value) {
    return L"\"" + value + L"\"";
}

static void ShowLaunchError(const std::wstring& message) {
    MessageBoxW(
        nullptr,
        message.c_str(),
        L"UVR5 ARM64 Launcher",
        MB_OK | MB_ICONERROR | MB_SETFOREGROUND
    );
}

int WINAPI wWinMain(
    HINSTANCE,
    HINSTANCE,
    PWSTR commandLineArgs,
    int
) {
    const std::wstring launcherPath = GetModulePath();
    if (launcherPath.empty()) {
        ShowLaunchError(L"Could not resolve launcher path.");
        return 1;
    }

    const std::wstring root = ParentDirectory(launcherPath);
    const std::wstring pythonw = root + L"\\.venv-arm64\\Scripts\\pythonw.exe";
    const std::wstring uvrScript = root + L"\\UVR.py";

    if (GetFileAttributesW(pythonw.c_str()) == INVALID_FILE_ATTRIBUTES) {
        ShowLaunchError(
            L"Native ARM64 Python environment was not found.\n\nExpected:\n" +
            pythonw +
            L"\n\nRun scripts\\bootstrap-win-arm64.ps1 first."
        );
        return 2;
    }

    if (GetFileAttributesW(uvrScript.c_str()) == INVALID_FILE_ATTRIBUTES) {
        ShowLaunchError(
            L"UVR.py was not found next to the launcher.\n\nExpected:\n" +
            uvrScript
        );
        return 3;
    }

    std::wstring command = Quote(pythonw) + L" " + Quote(uvrScript);

    if (commandLineArgs && commandLineArgs[0] != L'\0') {
        command += L" ";
        command += commandLineArgs;
    }

    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startupInfo{};
    startupInfo.cb = sizeof(startupInfo);

    PROCESS_INFORMATION processInfo{};

    const BOOL launched = CreateProcessW(
        pythonw.c_str(),
        mutableCommand.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
        nullptr,
        root.c_str(),
        &startupInfo,
        &processInfo
    );

    if (!launched) {
        const DWORD errorCode = GetLastError();
        ShowLaunchError(
            L"Failed to start UVR5 ARM64.\n\nWindows error: " +
            std::to_wstring(errorCode)
        );
        return 4;
    }

    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);
    return 0;
}
