//
//  DataPointsAIApp.swift
//  Data Points AI RSS Reader
//
//  Native macOS app wrapper for the FastAPI web server
//

import SwiftUI

@main
struct DataPointsAIApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var pythonServer = PythonServerManager.shared
    @StateObject private var appState = AppState.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 900, minHeight: 600)
                .environmentObject(pythonServer)
                .environmentObject(appState)
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                // Remove default "New Window" command
            }

            // File menu
            CommandGroup(after: .newItem) {
                Menu("Export") {
                    Button("Export as Markdown") {
                        appState.requestExport(format: .markdown)
                    }
                    .keyboardShortcut("e", modifiers: [.command, .shift])

                    Button("Export as Plain Text") {
                        appState.requestExport(format: .plainText)
                    }

                    Button("Export as JSON") {
                        appState.requestExport(format: .json)
                    }
                }

                Divider()
            }

            // Edit menu - Find
            CommandGroup(after: .pasteboard) {
                Divider()

                Button("Find...") {
                    appState.triggerFind()
                }
                .keyboardShortcut("f", modifiers: .command)

                Button("Find Next") {
                    appState.findNext()
                }
                .keyboardShortcut("g", modifiers: .command)

                Button("Find Previous") {
                    appState.findPrevious()
                }
                .keyboardShortcut("g", modifiers: [.command, .shift])
            }

            // View menu
            CommandMenu("View") {
                Button("Back") {
                    appState.webView?.goBack()
                }
                .keyboardShortcut("[", modifiers: .command)
                .disabled(!(appState.webView?.canGoBack ?? false))

                Button("Forward") {
                    appState.webView?.goForward()
                }
                .keyboardShortcut("]", modifiers: .command)
                .disabled(!(appState.webView?.canGoForward ?? false))

                Divider()

                Button("Reload") {
                    appState.reloadWebView()
                }
                .keyboardShortcut("r", modifiers: .command)

                Divider()

                Button("Toggle Sidebar") {
                    appState.toggleSidebar()
                }
                .keyboardShortcut("s", modifiers: [.command, .control])
            }

            // Bookmarks menu
            CommandMenu("Bookmarks") {
                Button("View All Bookmarks") {
                    appState.navigateTo(path: "/bookmarks")
                }
                .keyboardShortcut("b", modifiers: [.command, .shift])

                Divider()

                Menu("Recent Articles") {
                    ForEach(appState.recentArticles) { article in
                        Button(article.title) {
                            appState.openArticle(article)
                        }
                    }

                    if appState.recentArticles.isEmpty {
                        Text("No recent articles")
                            .disabled(true)
                    }
                }
            }

            // Help menu
            CommandGroup(replacing: .help) {
                Button("Data Points AI Help") {
                    NSWorkspace.shared.open(URL(string: "https://github.com/tcarmody/rss-reader")!)
                }
            }
        }

        Settings {
            SettingsView()
        }
    }
}
