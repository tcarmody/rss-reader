//
//  AppDelegate.swift
//  Data Points AI RSS Reader
//
//  Handles app lifecycle and Dock integration
//

import Cocoa
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusBarItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Start Python server
        PythonServerManager.shared.startServer()

        // Configure Dock menu
        NSApp.dockTile.badgeLabel = nil

        // Register for notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(updateDockBadge(_:)),
            name: NSNotification.Name("UpdateDockBadge"),
            object: nil
        )

        print("✅ Data Points AI RSS Reader launched")
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Stop Python server
        PythonServerManager.shared.stopServer()

        // Save state
        AppState.shared.saveState()

        print("👋 Data Points AI RSS Reader terminated")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        let dockMenu = NSMenu()

        dockMenu.addItem(withTitle: "Refresh Feeds", action: #selector(refreshFeeds), keyEquivalent: "")
        dockMenu.addItem(withTitle: "View Bookmarks", action: #selector(viewBookmarks), keyEquivalent: "")

        dockMenu.addItem(NSMenuItem.separator())

        // Recent articles
        let recentArticles = AppState.shared.recentArticles.prefix(5)
        if !recentArticles.isEmpty {
            for article in recentArticles {
                let menuItem = NSMenuItem(
                    title: article.title,
                    action: #selector(openRecentArticle(_:)),
                    keyEquivalent: ""
                )
                menuItem.representedObject = article
                dockMenu.addItem(menuItem)
            }
        } else {
            let item = NSMenuItem(title: "No recent articles", action: nil, keyEquivalent: "")
            item.isEnabled = false
            dockMenu.addItem(item)
        }

        return dockMenu
    }

    @objc func updateDockBadge(_ notification: Notification) {
        guard let count = notification.object as? Int else { return }

        DispatchQueue.main.async {
            if count > 0 {
                NSApp.dockTile.badgeLabel = "\(count)"
            } else {
                NSApp.dockTile.badgeLabel = nil
            }
        }
    }

    @objc func refreshFeeds() {
        AppState.shared.refreshFeeds()
    }

    @objc func viewBookmarks() {
        AppState.shared.navigateTo(path: "/bookmarks")
    }

    @objc func openRecentArticle(_ sender: NSMenuItem) {
        guard let article = sender.representedObject as? Article else { return }
        AppState.shared.openArticle(article)
    }
}
