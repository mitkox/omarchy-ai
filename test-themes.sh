#!/bin/bash

# Simple test script to verify themes installation works
echo "🧪 Testing themes installation..."

# Create temporary test directories
TEST_DIR="/tmp/omarchy-test"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR/.local/share/omarchy-ai"
mkdir -p "$TEST_DIR/.local/share/omarchy"
mkdir -p "$TEST_DIR/.config/omarchy"

# Copy current themes to test directory
cp -r themes "$TEST_DIR/.local/share/omarchy-ai/"

# Set HOME to test directory for this session
export HOME="$TEST_DIR"

# Test the theme installation script
echo "📂 Testing theme.sh script..."
if source install/theme.sh; then
    echo "✅ theme.sh executed successfully"
    
    # Check if themes were linked correctly
    if [ -d "$HOME/.config/omarchy/themes" ]; then
        echo "✅ Themes directory created"
        theme_count=$(find "$HOME/.config/omarchy/themes" -maxdepth 1 -type l -o -type d | grep -v '^\.$' | wc -l)
        echo "📊 Found $theme_count themes linked"
    else
        echo "❌ Themes directory not created"
    fi
    
    # Check if current theme was set
    if [ -L "$HOME/.config/omarchy/current/theme" ]; then
        current_theme=$(basename "$(readlink "$HOME/.config/omarchy/current/theme")")
        echo "✅ Current theme set to: $current_theme"
    else
        echo "❌ Current theme not set"
    fi
    
else
    echo "❌ theme.sh failed to execute"
fi

# Test backgrounds script
echo "📂 Testing backgrounds.sh script..."
if source install/backgrounds.sh; then
    echo "✅ backgrounds.sh executed successfully"
else
    echo "❌ backgrounds.sh failed to execute"
fi

# Cleanup
echo "🧹 Cleaning up test environment..."
rm -rf "$TEST_DIR"

echo "🎉 Theme installation test completed!"