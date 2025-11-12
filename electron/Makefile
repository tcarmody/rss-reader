# Makefile for Data Points AI RSS Reader Mac App

.PHONY: help install dev build build-universal build-arm64 build-x64 clean icon

# Default target
help:
	@echo "Data Points AI RSS Reader - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make install         - Install all dependencies"
	@echo "  make dev            - Run in development mode"
	@echo "  make build          - Build for all architectures"
	@echo "  make build-universal - Build universal binary (recommended)"
	@echo "  make build-arm64    - Build for Apple Silicon only"
	@echo "  make build-x64      - Build for Intel only"
	@echo "  make icon           - Generate app icon (requires source image)"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make dist-clean     - Clean everything including node_modules"

# Install dependencies
install:
	@echo "Installing Node.js dependencies..."
	npm install
	@echo ""
	@echo "Checking Python dependencies..."
	@cd .. && ./run_server.sh --help > /dev/null || echo "Run ../run_server.sh to set up Python"
	@echo ""
	@echo "✓ Installation complete"

# Run in development mode
dev:
	npm run dev

# Build for all architectures
build:
	./build.sh all

# Build universal binary
build-universal:
	./build.sh universal

# Build for Apple Silicon
build-arm64:
	./build.sh arm64

# Build for Intel
build-x64:
	./build.sh x64

# Generate icon
icon:
	@if [ -f "icon-source.png" ]; then \
		./create-icons.sh icon-source.png; \
	elif [ -f "../static/icon.png" ]; then \
		./create-icons.sh ../static/icon.png; \
	else \
		echo "No source icon found. Place a 1024x1024 PNG at icon-source.png"; \
		./create-icons.sh; \
	fi

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf out/
	rm -rf build/
	@echo "✓ Build artifacts cleaned"

# Clean everything
dist-clean: clean
	rm -rf node_modules/
	rm -f package-lock.json
	@echo "✓ All build files cleaned"

# Quick rebuild
rebuild: clean build-universal
