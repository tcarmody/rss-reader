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

    var body: some View {
        VStack(spacing: 0) {
            // Native toolbar
            NativeToolbar()
                .environmentObject(pythonServer)
                .environmentObject(appState)

            // Status bar (only shown when server is not running)
            if pythonServer.serverStatus != .running {
                ServerStatusBar(status: pythonServer.serverStatus, errorMessage: pythonServer.errorMessage)
            }

            // Main web view
            WebView()
                .environmentObject(pythonServer)
                .environmentObject(appState)
        }
    }
}

struct NativeToolbar: View {
    @EnvironmentObject var pythonServer: PythonServerManager
    @EnvironmentObject var appState: AppState
    @State private var canGoBack = false
    @State private var canGoForward = false
    @State private var pulseAnimation = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                // Browser-style navigation
                HStack(spacing: 4) {
                    Button(action: { appState.webView?.goBack() }) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .buttonStyle(.borderless)
                    .disabled(!canGoBack)
                    .help("Go back (⌘[)")

                    Button(action: { appState.webView?.goForward() }) {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 14, weight: .medium))
                    }
                    .buttonStyle(.borderless)
                    .disabled(!canGoForward)
                    .help("Go forward (⌘])")
                }

                Divider()
                    .frame(height: 24)

                // Quick navigation
                Button(action: { appState.navigateTo(path: "/") }) {
                    Image(systemName: "house.fill")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Home")

                Button(action: { appState.navigateTo(path: "/bookmarks") }) {
                    Image(systemName: "bookmark.fill")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Bookmarks (⌘⇧B)")

                Divider()
                    .frame(height: 24)

                // Page title and URL
                VStack(alignment: .leading, spacing: 2) {
                    Text(pageTitle)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                        .foregroundColor(.primary)

                    if let url = appState.currentURL, !url.isEmpty {
                        Text(displayURL(url))
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }
                .frame(maxWidth: 300, alignment: .leading)

                Spacer()

                // Server status indicator with animation
                HStack(spacing: 8) {
                    ZStack {
                        if pythonServer.serverStatus == .starting {
                            Circle()
                                .fill(serverStatusColor.opacity(0.3))
                                .frame(width: 16, height: 16)
                                .scaleEffect(pulseAnimation ? 1.5 : 1.0)
                                .opacity(pulseAnimation ? 0 : 1)
                                .animation(.easeInOut(duration: 1.0).repeatForever(autoreverses: false), value: pulseAnimation)
                        }

                        Circle()
                            .fill(serverStatusColor)
                            .frame(width: 8, height: 8)
                    }

                    Text(serverStatusText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .onAppear {
                    if pythonServer.serverStatus == .starting {
                        pulseAnimation = true
                    }
                }

                Divider()
                    .frame(height: 24)

                // Feature buttons
                Button(action: { appState.navigateTo(path: "/summarize") }) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Summarize URL")

                Button(action: { appState.navigateTo(path: "/feeds") }) {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Manage Feeds")

                Divider()
                    .frame(height: 24)

                // Action buttons
                Button(action: { appState.reloadWebView() }) {
                    if appState.isLoading {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 13, height: 13)
                    } else {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 13))
                    }
                }
                .buttonStyle(.bordered)
                .help(appState.isLoading ? "Loading..." : "Refresh (⌘R)")
                .disabled(appState.isLoading)

                Menu {
                    Button("Markdown (⌘⇧E)") {
                        appState.requestExport(format: .markdown)
                    }
                    Button("Plain Text") {
                        appState.requestExport(format: .plainText)
                    }
                    Button("JSON") {
                        appState.requestExport(format: .json)
                    }
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Export articles")

                Button(action: { appState.triggerFind() }) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Find in page (⌘F)")

                Button(action: {
                    appState.navigateTo(path: "/settings")
                }) {
                    Image(systemName: "gearshape")
                        .font(.system(size: 13))
                }
                .buttonStyle(.bordered)
                .help("Settings (⌘,)")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()
        }
        .background(
            Rectangle()
                .fill(Color(NSColor.controlBackgroundColor))
                .shadow(color: Color.black.opacity(0.05), radius: 2, y: 1)
        )
        .onChange(of: appState.currentURL) { _ in
            updateNavigationState()
        }
        .onAppear {
            updateNavigationState()
        }
    }

    private var pageTitle: String {
        if let url = appState.currentURL {
            if url.contains("/bookmarks") {
                return "Bookmarks"
            } else if url.contains("/summarize") {
                return "Article Summary"
            } else if url == "http://127.0.0.1:5005/" || url == "http://127.0.0.1:5005" {
                return "RSS Feed"
            } else {
                return "Data Points AI"
            }
        }
        return "Data Points AI"
    }

    private func displayURL(_ url: String) -> String {
        return url
            .replacingOccurrences(of: "http://127.0.0.1:5005", with: "")
            .replacingOccurrences(of: "http://", with: "")
            .replacingOccurrences(of: "https://", with: "")
    }

    private func updateNavigationState() {
        canGoBack = appState.webView?.canGoBack ?? false
        canGoForward = appState.webView?.canGoForward ?? false
    }

    private var serverStatusColor: Color {
        switch pythonServer.serverStatus {
        case .running:
            return .green
        case .starting:
            return .orange
        case .stopped:
            return .gray
        case .error:
            return .red
        }
    }

    private var serverStatusText: String {
        switch pythonServer.serverStatus {
        case .running:
            return "Connected"
        case .starting:
            return "Starting..."
        case .stopped:
            return "Disconnected"
        case .error:
            return "Error"
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

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()

        // Enable developer extras in debug builds
        #if DEBUG
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        #endif

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator

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
            DispatchQueue.main.async {
                self.parent.appState.isLoading = true
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("✅ Page loaded: \(webView.url?.absoluteString ?? "unknown")")

            // Inject JavaScript bridge
            injectJavaScriptBridge(webView: webView)

            // Update app state
            DispatchQueue.main.async {
                self.parent.appState.currentURL = webView.url?.absoluteString
                self.parent.appState.isLoading = false
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            print("❌ Page load failed: \(error.localizedDescription)")

            DispatchQueue.main.async {
                self.parent.appState.isLoading = false
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("❌ Provisional navigation failed: \(error.localizedDescription)")

            DispatchQueue.main.async {
                self.parent.appState.isLoading = false
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
