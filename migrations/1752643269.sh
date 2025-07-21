echo "Add new matte black theme"

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

# Create themes directory if it doesn't exist
mkdir -p ~/.config/omarchy/themes

# Check for matte-black theme in omarchy-ai first, then base omarchy
matte_black_theme=""
if [ -d ~/.local/share/omarchy-ai/themes/matte-black ]; then
  matte_black_theme="~/.local/share/omarchy-ai/themes/matte-black"
elif [ -d ~/.local/share/omarchy/themes/matte-black ]; then
  matte_black_theme="~/.local/share/omarchy/themes/matte-black"
fi

if [[ -n "$matte_black_theme" ]] && [[ ! -L ~/.config/omarchy/themes/matte-black ]]; then
  ln -sfn "$matte_black_theme" ~/.config/omarchy/themes/matte-black
  if [ -f "$matte_black_theme/backgrounds.sh" ]; then
    source "$matte_black_theme/backgrounds.sh" 2>/dev/null || echo "⚠️  Could not process matte-black backgrounds"
  fi
  echo "✅ Matte black theme added"
else
  echo "⚠️  Matte black theme not found or already linked"
fi
