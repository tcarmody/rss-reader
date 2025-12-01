//
//  PythonServerManager.swift
//  Data Points AI RSS Reader
//
//  Manages the Python FastAPI server lifecycle
//

import Foundation
import Combine

final class PythonServerManager: ObservableObject {
    static let shared = PythonServerManager()

    @Published var isRunning = false
    @Published var serverURL = "http://127.0.0.1:5005"
    @Published var serverStatus: ServerStatus = .stopped
    @Published var errorMessage: String?

    private var pythonProcess: Process?
    private var healthCheckTimer: Timer?
    private let serverPort = 5005

    enum ServerStatus {
        case stopped
        case starting
        case running
        case error
    }

    private init() {}

    func startServer() {
        guard pythonProcess == nil else {
            print("⚠️ Python server already running")
            return
        }

        serverStatus = .starting
        print("🚀 Starting Python server...")

        // Find Python executable
        guard let pythonPath = findPython() else {
            errorMessage = "Could not find Python executable"
            serverStatus = .error
            print("❌ \(errorMessage!)")
            return
        }

        print("🐍 Using Python at: \(pythonPath)")

        // Get project path
        guard let projectPath = getProjectPath() else {
            errorMessage = "Could not find project directory"
            serverStatus = .error
            print("❌ \(errorMessage!)")
            return
        }

        print("📁 Project path: \(projectPath)")

        // Create Python process
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [
            "-m", "uvicorn",
            "server:app",
            "--host", "127.0.0.1",
            "--port", "\(serverPort)"
        ]
        process.currentDirectoryURL = URL(fileURLWithPath: projectPath)

        // Set environment
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        // Capture output
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        // Read output
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                print("📝 Python: \(output.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        errorPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    print("⚠️ Python error: \(trimmed)")
                }
            }
        }

        // Handle termination
        process.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.serverStatus = .stopped
                self?.stopHealthCheck()

                if process.terminationStatus != 0 {
                    self?.errorMessage = "Python server exited with code \(process.terminationStatus)"
                    print("❌ \(self?.errorMessage ?? "")")
                } else {
                    print("✅ Python server stopped gracefully")
                }
            }
        }

        do {
            try process.run()
            pythonProcess = process

            // Wait a bit for server to start, then check health
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                self?.checkHealth()
                self?.startHealthCheck()
            }

            print("✅ Python process started (PID: \(process.processIdentifier))")
        } catch {
            errorMessage = "Failed to start Python server: \(error.localizedDescription)"
            serverStatus = .error
            print("❌ \(errorMessage!)")
        }
    }

    func stopServer() {
        guard let process = pythonProcess else { return }

        print("🛑 Stopping Python server...")

        stopHealthCheck()

        process.terminate()

        // Force kill if not terminated within 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
            if process.isRunning {
                print("⚠️ Force killing Python server")
                process.interrupt()
            }
            self?.pythonProcess = nil
        }
    }

    private func findPython() -> String? {
        // Try to find Python in order of preference
        let paths = [
            // Virtual environment in project
            "../rss_venv/bin/python",
            "../venv/bin/python",

            // System Python locations
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3"
        ]

        // Get bundle resource path
        let bundlePath = Bundle.main.resourcePath ?? ""

        for path in paths {
            var fullPath: String

            if path.hasPrefix("/") {
                // Absolute path
                fullPath = path
            } else {
                // Relative to bundle
                fullPath = (bundlePath as NSString).appendingPathComponent(path)
            }

            // Resolve symlinks and check existence
            if let resolvedPath = try? FileManager.default.destinationOfSymbolicLink(atPath: fullPath) {
                if FileManager.default.fileExists(atPath: resolvedPath) {
                    return resolvedPath
                }
            } else if FileManager.default.fileExists(atPath: fullPath) {
                return fullPath
            }
        }

        // Fallback: search PATH
        return findInPath("python3")
    }

    private func findInPath(_ executable: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [executable]

        let pipe = Pipe()
        process.standardOutput = pipe

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
               !path.isEmpty {
                return path
            }
        } catch {
            print("⚠️ Error finding \(executable) in PATH: \(error)")
        }

        return nil
    }

    private func getProjectPath() -> String? {
        // In development, project is in parent of macos-app
        // In production, it's bundled in Resources/app
        let bundleResourcePath = Bundle.main.resourcePath ?? ""

        let paths = [
            // Production: bundled in Resources/app
            (bundleResourcePath as NSString).appendingPathComponent("app"),

            // Development: Try multiple locations
            // When running from Xcode, bundle is in DerivedData/.../Contents/Resources
            // Project root is /Users/timcarmody/workspace/rss-reader
            ((bundleResourcePath as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent,

            // Go up to macos-app/Data Points AI.app/Contents/Resources
            // Then up 4 levels: Resources -> Contents -> Data Points AI.app -> macos-app -> rss-reader
            URL(fileURLWithPath: bundleResourcePath)
                .deletingLastPathComponent()  // Remove Resources
                .deletingLastPathComponent()  // Remove Contents
                .deletingLastPathComponent()  // Remove Data Points AI.app
                .deletingLastPathComponent()  // Remove macos-app (or DerivedData path)
                .path,

            // Absolute fallback for development
            "/Users/timcarmody/workspace/rss-reader"
        ]

        for path in paths {
            // Check if server.py exists
            let serverPath = (path as NSString).appendingPathComponent("server.py")
            if FileManager.default.fileExists(atPath: serverPath) {
                print("✅ Found project at: \(path)")
                return path
            } else {
                print("⚠️ Checked path: \(path) - server.py not found")
            }
        }

        return nil
    }

    private func checkHealth() {
        guard let url = URL(string: "\(serverURL)/status") else { return }

        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    self?.isRunning = true
                    self?.serverStatus = .running
                    self?.errorMessage = nil
                    print("✅ Python server health check: OK")
                } else {
                    if self?.serverStatus == .starting {
                        // Still starting, keep waiting
                        print("⏳ Server still starting...")
                    } else {
                        self?.serverStatus = .error
                        self?.errorMessage = "Server not responding"
                        print("❌ Server health check failed")
                    }
                }
            }
        }.resume()
    }

    private func startHealthCheck() {
        stopHealthCheck()

        healthCheckTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            self?.checkHealth()
        }
    }

    private func stopHealthCheck() {
        healthCheckTimer?.invalidate()
        healthCheckTimer = nil
    }

    deinit {
        stopServer()
    }
}
