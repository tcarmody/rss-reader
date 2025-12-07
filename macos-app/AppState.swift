//
//  AppState.swift
//  Data Points AI RSS Reader
//
//  Application state management and persistence
//

import Foundation
import SwiftUI
import WebKit
import Combine

final class AppState: ObservableObject {
    static let shared = AppState()

    // Web view reference
    weak var webView: WKWebView?

    // Navigation
    @Published var currentURL: String?
    @Published var pendingNavigation: String?
    @Published var shouldReload = false

    // Find in page
    @Published var shouldTriggerFind = false
    @Published var findQuery = ""
    @Published var isFindPanelVisible = false

    // Recent articles
    @Published var recentArticles: [Article] = []

    // Export
    @Published var exportFormat: ExportFormat?

    // Window state
    @Published var windowFrame: NSRect?
    @Published var isMaximized = false

    // Unread count for badge
    @Published var unreadCount = 0 {
        didSet {
            updateDockBadge()
        }
    }

    enum ExportFormat {
        case markdown
        case plainText
        case json
    }

    private let defaults = UserDefaults.standard
    private let recentArticlesKey = "recentArticles"
    private let windowFrameKey = "windowFrame"
    private let isMaximizedKey = "isMaximized"
    private let maxRecentArticles = 10

    private init() {
        loadState()
    }

    // MARK: - Navigation

    func navigateTo(path: String) {
        pendingNavigation = path
    }

    func reloadWebView() {
        shouldReload = true
    }

    func toggleSidebar() {
        // Execute JavaScript to toggle sidebar if it exists
        webView?.evaluateJavaScript("if (window.toggleSidebar) window.toggleSidebar();")
    }

    // MARK: - Find in Page

    func triggerFind() {
        shouldTriggerFind = true
    }

    func showFindPanel() {
        isFindPanelVisible = true
    }

    func findNext() {
        webView?.evaluateJavaScript("if (window.findNext) window.findNext();")
    }

    func findPrevious() {
        webView?.evaluateJavaScript("if (window.findPrevious) window.findPrevious();")
    }

    func performFind(query: String) {
        findQuery = query
        let jsQuery = query.replacingOccurrences(of: "\"", with: "\\\"")
        webView?.evaluateJavaScript("if (window.performFind) window.performFind('\(jsQuery)');")
    }

    // MARK: - Articles

    func openArticle(_ article: Article) {
        navigateTo(path: "/article/\(article.id)")
        addRecentArticle(article)
    }

    func addRecentArticle(_ article: Article) {
        // Remove if already exists
        recentArticles.removeAll { $0.id == article.id }

        // Add to beginning
        recentArticles.insert(article, at: 0)

        // Limit size
        if recentArticles.count > maxRecentArticles {
            recentArticles = Array(recentArticles.prefix(maxRecentArticles))
        }

        saveRecentArticles()
    }

    func refreshFeeds() {
        webView?.evaluateJavaScript("if (window.refreshFeeds) window.refreshFeeds();")
    }

    // MARK: - Export

    func requestExport(format: ExportFormat) {
        exportFormat = format

        // Show save panel
        let savePanel = NSSavePanel()
        savePanel.canCreateDirectories = true
        savePanel.showsTagField = false

        switch format {
        case .markdown:
            savePanel.allowedContentTypes = [.init(filenameExtension: "md")!]
            savePanel.nameFieldStringValue = "articles.md"
        case .plainText:
            savePanel.allowedContentTypes = [.plainText]
            savePanel.nameFieldStringValue = "articles.txt"
        case .json:
            savePanel.allowedContentTypes = [.json]
            savePanel.nameFieldStringValue = "articles.json"
        }

        savePanel.begin { [weak self] response in
            guard response == .OK, let url = savePanel.url else { return }
            self?.exportArticles(to: url, format: format)
        }
    }

    private func exportArticles(to url: URL, format: ExportFormat) {
        // Request export data from web view
        let formatString = switch format {
        case .markdown: "md"
        case .plainText: "txt"
        case .json: "json"
        }

        let script = "if (window.getExportData) window.getExportData('\(formatString)');"

        webView?.evaluateJavaScript(script) { result, error in
            if let error = error {
                print("❌ Export failed: \(error)")
                return
            }

            guard let content = result as? String else {
                print("❌ No export data returned")
                return
            }

            do {
                try content.write(to: url, atomically: true, encoding: .utf8)
                print("✅ Exported to \(url.path)")

                // Show in Finder
                NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: "")
            } catch {
                print("❌ Failed to write export file: \(error)")
            }
        }
    }

    // MARK: - Dock Integration

    private func updateDockBadge() {
        NotificationCenter.default.post(
            name: NSNotification.Name("UpdateDockBadge"),
            object: unreadCount
        )
    }

    // MARK: - State Persistence

    func saveState() {
        saveRecentArticles()
        saveWindowState()
    }

    func loadState() {
        loadRecentArticles()
        loadWindowState()
    }

    private func saveRecentArticles() {
        if let encoded = try? JSONEncoder().encode(recentArticles) {
            defaults.set(encoded, forKey: recentArticlesKey)
        }
    }

    private func loadRecentArticles() {
        if let data = defaults.data(forKey: recentArticlesKey),
           let decoded = try? JSONDecoder().decode([Article].self, from: data) {
            recentArticles = decoded
        }
    }

    private func saveWindowState() {
        if let frame = windowFrame {
            let dict: [String: CGFloat] = [
                "x": frame.origin.x,
                "y": frame.origin.y,
                "width": frame.size.width,
                "height": frame.size.height
            ]
            defaults.set(dict, forKey: windowFrameKey)
        }
        defaults.set(isMaximized, forKey: isMaximizedKey)
    }

    private func loadWindowState() {
        if let dict = defaults.dictionary(forKey: windowFrameKey) as? [String: CGFloat],
           let x = dict["x"], let y = dict["y"],
           let width = dict["width"], let height = dict["height"] {
            windowFrame = NSRect(x: x, y: y, width: width, height: height)
        }
        isMaximized = defaults.bool(forKey: isMaximizedKey)
    }
}

// MARK: - Article Model

struct Article: Identifiable, Codable {
    let id: String
    let title: String
    let url: String?
    let summary: String?
    let date: Date?

    init(id: String, title: String, url: String? = nil, summary: String? = nil, date: Date? = nil) {
        self.id = id
        self.title = title
        self.url = url
        self.summary = summary
        self.date = date
    }
}
