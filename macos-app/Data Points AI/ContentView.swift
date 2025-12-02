//
//  ContentView.swift
//  Data Points AI RSS Reader
//
//  Main view with WKWebView integration
//

import SwiftUI
import WebKit

struct ContentView: View {
    @EnvironmentObject var pythonServer: PythonServerManager
    @EnvironmentObject var appState: AppState
    @State private var isLoading = false

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                // Status bar (only shown when server is not running)
                if pythonServer.serverStatus != .running {
                    ServerStatusBar(status: pythonServer.serverStatus, errorMessage: pythonServer.errorMessage)
                }

                // Main web view
                WebView(isLoading: $isLoading)
                    .environmentObject(pythonServer)
                    .environmentObject(appState)
            }

            // Loading overlay for long operations
            if isLoading {
                LoadingOverlay()
            }
        }
    }
}

struct LoadingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.5)
                    .progressViewStyle(.circular)

                Text("Generating summary...")
                    .font(.headline)
                    .foregroundColor(.white)

                Text("This may take 1-3 minutes for long articles")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.8))
            }
            .padding(40)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(NSColor.controlBackgroundColor).opacity(0.95))
            )
            .shadow(radius: 20)
        }
    }
}

struct ServerStatusBar: View {
    let status: PythonServerManager.ServerStatus
    let errorMessage: String?

    var body: some View {
        HStack {
            Image(systemName: statusIcon)
                .foregroundColor(statusColor)

            Text(statusText)
                .font(.system(size: 13))

            Spacer()

            if status == .starting {
                ProgressView()
                    .scaleEffect(0.7)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(statusBackground)
    }

    private var statusIcon: String {
        switch status {
        case .stopped:
            return "circle.fill"
        case .starting:
            return "arrow.clockwise.circle.fill"
        case .running:
            return "checkmark.circle.fill"
        case .error:
            return "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch status {
        case .stopped:
            return .gray
        case .starting:
            return .blue
        case .running:
            return .green
        case .error:
            return .red
        }
    }

    private var statusText: String {
        switch status {
        case .stopped:
            return "Server stopped"
        case .starting:
            return "Starting server..."
        case .running:
            return "Server running"
        case .error:
            return errorMessage ?? "Server error"
        }
    }

    private var statusBackground: Color {
        switch status {
        case .error:
            return Color.red.opacity(0.1)
        case .starting:
            return Color.blue.opacity(0.1)
        default:
            return Color.gray.opacity(0.1)
        }
    }
}

struct WebView: NSViewRepresentable {
    @EnvironmentObject var pythonServer: PythonServerManager
    @EnvironmentObject var appState: AppState
    @Binding var isLoading: Bool

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()

        // Enable developer extras in debug builds
        #if DEBUG
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        #endif

        // Set longer timeout for API requests (5 minutes for long articles)
        configuration.preferences.setValue(true, forKey: "allowsInlineMediaPlayback")

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator

        // Increase timeout for long-running requests
        webView.configuration.processPool.setValue(300, forKey: "_maximumSuspensionTime")

        // Store web view reference in app state
        DispatchQueue.main.async {
            appState.webView = webView
        }

        // Load the server URL
        if pythonServer.isRunning {
            loadURL(webView: webView)
        }

        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        // Reload when server becomes running
        if pythonServer.isRunning && webView.url == nil {
            loadURL(webView: webView)
        }

        // Handle reload requests
        if appState.shouldReload {
            webView.reload()
            appState.shouldReload = false
        }

        // Handle navigation requests
        if let path = appState.pendingNavigation {
            navigateTo(webView: webView, path: path)
            appState.pendingNavigation = nil
        }

        // Handle find requests
        if appState.shouldTriggerFind {
            appState.showFindPanel()
            appState.shouldTriggerFind = false
        }
    }

    private func loadURL(webView: WKWebView) {
        guard let url = URL(string: pythonServer.serverURL) else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 300 // 5 minutes timeout for long articles
        webView.load(request)
    }

    private func navigateTo(webView: WKWebView, path: String) {
        guard let url = URL(string: "\(pythonServer.serverURL)\(path)") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 300 // 5 minutes timeout for long articles
        webView.load(request)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        var parent: WebView

        init(_ parent: WebView) {
            self.parent = parent
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            // Show loading indicator when navigation starts
            DispatchQueue.main.async {
                self.parent.isLoading = true
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("✅ Page loaded: \(webView.url?.absoluteString ?? "unknown")")

            // Hide loading indicator
            DispatchQueue.main.async {
                self.parent.isLoading = false
            }

            // Inject JavaScript bridge
            injectJavaScriptBridge(webView: webView)

            // Update app state
            DispatchQueue.main.async {
                self.parent.appState.currentURL = webView.url?.absoluteString
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            print("❌ Page load failed: \(error.localizedDescription)")

            // Hide loading indicator
            DispatchQueue.main.async {
                self.parent.isLoading = false
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("❌ Provisional navigation failed: \(error.localizedDescription)")

            // Hide loading indicator
            DispatchQueue.main.async {
                self.parent.isLoading = false
            }

            // If server not ready, retry after a delay
            if !self.parent.pythonServer.isRunning {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    if self.parent.pythonServer.isRunning {
                        self.parent.loadURL(webView: webView)
                    }
                }
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            // Handle external links
            if let url = navigationAction.request.url,
               url.scheme == "http" || url.scheme == "https" {

                // If it's an external link (not our server), open in default browser
                if !url.absoluteString.starts(with: self.parent.pythonServer.serverURL) {
                    NSWorkspace.shared.open(url)
                    decisionHandler(.cancel)
                    return
                }
            }

            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
            // Handle links with target="_blank" - open in default browser
            if let url = navigationAction.request.url {
                NSWorkspace.shared.open(url)
            }
            return nil
        }

        private func injectJavaScriptBridge(webView: WKWebView) {
            // Inject a bridge to detect we're running in native app
            let script = """
            window.isNativeApp = true;
            window.isMacApp = true;

            // Override badge update to communicate with Swift
            window.updateNativeBadge = function(count) {
                // This would use a proper message handler in production
                console.log('Badge update:', count);
            };

            console.log('🍎 Running in native macOS app');
            """

            webView.evaluateJavaScript(script) { result, error in
                if let error = error {
                    print("⚠️ JavaScript injection error: \(error)")
                }
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(PythonServerManager.shared)
        .environmentObject(AppState.shared)
        .frame(width: 1200, height: 800)
}
