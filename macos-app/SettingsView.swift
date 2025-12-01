//
//  SettingsView.swift
//  Data Points AI RSS Reader
//
//  Application settings panel
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("serverPort") private var serverPort = 5005
    @AppStorage("autoRefresh") private var autoRefresh = false
    @AppStorage("autoRefreshInterval") private var autoRefreshInterval = 30.0
    @AppStorage("showNotifications") private var showNotifications = true
    @AppStorage("defaultExportFormat") private var defaultExportFormat = "markdown"

    var body: some View {
        TabView {
            GeneralSettingsView(
                autoRefresh: $autoRefresh,
                autoRefreshInterval: $autoRefreshInterval,
                showNotifications: $showNotifications
            )
            .tabItem {
                Label("General", systemImage: "gear")
            }

            ServerSettingsView(serverPort: $serverPort)
                .tabItem {
                    Label("Server", systemImage: "server.rack")
                }

            ExportSettingsView(defaultFormat: $defaultExportFormat)
                .tabItem {
                    Label("Export", systemImage: "square.and.arrow.up")
                }

            AboutView()
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
        }
        .frame(width: 500, height: 400)
    }
}

struct GeneralSettingsView: View {
    @Binding var autoRefresh: Bool
    @Binding var autoRefreshInterval: Double
    @Binding var showNotifications: Bool

    var body: some View {
        Form {
            Section {
                Toggle("Automatically refresh feeds", isOn: $autoRefresh)

                if autoRefresh {
                    HStack {
                        Text("Refresh every")
                        Slider(value: $autoRefreshInterval, in: 5...120, step: 5)
                        Text("\(Int(autoRefreshInterval)) minutes")
                            .frame(width: 80, alignment: .trailing)
                    }
                }
            } header: {
                Text("Feed Refresh")
            }

            Section {
                Toggle("Show notifications for new articles", isOn: $showNotifications)
            } header: {
                Text("Notifications")
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct ServerSettingsView: View {
    @Binding var serverPort: Int

    var body: some View {
        Form {
            Section {
                HStack {
                    Text("Port:")
                    TextField("Port", value: $serverPort, format: .number)
                        .frame(width: 80)
                    Text("(requires restart)")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }

                Text("The Python server will run on http://127.0.0.1:\(serverPort)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } header: {
                Text("Server Configuration")
            }

            Section {
                Text("Note: Changing the server port requires restarting the application.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct ExportSettingsView: View {
    @Binding var defaultFormat: String

    var body: some View {
        Form {
            Section {
                Picker("Default export format:", selection: $defaultFormat) {
                    Text("Markdown").tag("markdown")
                    Text("Plain Text").tag("text")
                    Text("JSON").tag("json")
                }
                .pickerStyle(.radioGroup)
            } header: {
                Text("Export Settings")
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct AboutView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "book.fill")
                .font(.system(size: 64))
                .foregroundColor(.blue)

            Text("Data Points AI RSS Reader")
                .font(.title)
                .fontWeight(.bold)

            Text("Version 1.0.0")
                .font(.subheadline)
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                Text("A native macOS app for reading and analyzing RSS feeds")
                    .multilineTextAlignment(.center)
                    .font(.body)

                Text("Powered by Claude AI")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            VStack(spacing: 12) {
                Link("View on GitHub", destination: URL(string: "https://github.com/tcarmody/rss-reader")!)
                    .font(.body)

                Text("© 2025 Data Points AI")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#Preview {
    SettingsView()
}
