BACKGROUNDS_DIR=~/.config/omarchy/backgrounds/

download_background_image() {
  local url="$1"
  local path="$2"
  mkdir -p "$(dirname "$BACKGROUNDS_DIR/$path")"
  if command -v gum >/dev/null 2>&1; then
    gum spin --title "Downloading $url as $path..." -- curl -sL -o "$BACKGROUNDS_DIR/$path" "$url"
  else
    echo "📥 Downloading background: $path..."
    curl -sL -o "$BACKGROUNDS_DIR/$path" "$url"
  fi
}

# Download backgrounds from omarchy-ai themes first
if [ -d ~/.local/share/omarchy-ai/themes ]; then
  echo "📂 Processing omarchy-ai theme backgrounds..."
  for t in ~/.local/share/omarchy-ai/themes/*; do 
    if [ -d "$t" ] && [ -f "$t/backgrounds.sh" ]; then
      echo "  Processing $(basename "$t") backgrounds..."
      source "$t/backgrounds.sh" 2>/dev/null || echo "⚠️  Could not process backgrounds for $(basename "$t")"
    fi
  done
fi

# Process base omarchy themes as fallback
if [ -d ~/.local/share/omarchy/themes ]; then
  echo "📂 Processing base omarchy theme backgrounds..."
  for t in ~/.local/share/omarchy/themes/*; do 
    if [ -d "$t" ] && [ -f "$t/backgrounds.sh" ]; then
      echo "  Processing $(basename "$t") backgrounds..."
      source "$t/backgrounds.sh" 2>/dev/null || echo "⚠️  Could not process backgrounds for $(basename "$t")"
    fi
  done
fi

# Verify backgrounds directory was created
if [ ! -d "$BACKGROUNDS_DIR" ]; then
  echo "⚠️  No backgrounds were downloaded, creating empty directory"
  mkdir -p "$BACKGROUNDS_DIR"
fi
