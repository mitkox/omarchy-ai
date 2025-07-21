# Use dark mode for QT apps too (like kdenlive)
sudo pacman -S --noconfirm kvantum-qt5

# Prefer dark mode everything
sudo pacman -S --noconfirm gnome-themes-extra # Adds Adwaita-dark theme
gsettings set org.gnome.desktop.interface gtk-theme "Adwaita-dark" 2>/dev/null || echo "⚠️  Could not set GTK theme (gsettings not available)"
gsettings set org.gnome.desktop.interface color-scheme "prefer-dark" 2>/dev/null || echo "⚠️  Could not set color scheme (gsettings not available)"

# Setup theme links - prioritize omarchy-ai themes, fallback to base omarchy themes
mkdir -p ~/.config/omarchy/themes

# First, link themes from omarchy-ai if they exist
if [ -d ~/.local/share/omarchy-ai/themes ]; then
  echo "📂 Linking omarchy-ai themes..."
  for f in ~/.local/share/omarchy-ai/themes/*; do 
    if [ -d "$f" ]; then
      theme_name=$(basename "$f")
      ln -sfn "$f" ~/.config/omarchy/themes/"$theme_name"
    fi
  done
fi

# Fallback to base omarchy themes if they exist
if [ -d ~/.local/share/omarchy/themes ]; then
  echo "📂 Linking base omarchy themes..."
  for f in ~/.local/share/omarchy/themes/*; do 
    if [ -d "$f" ]; then
      theme_name=$(basename "$f")
      # Only link if we don't already have this theme from omarchy-ai
      if [ ! -L ~/.config/omarchy/themes/"$theme_name" ]; then
        ln -sfn "$f" ~/.config/omarchy/themes/"$theme_name"
      fi
    fi
  done
fi

# Set initial theme - try tokyo-night first, fallback to first available theme
mkdir -p ~/.config/omarchy/current

if [ -d ~/.config/omarchy/themes/tokyo-night ]; then
  echo "🎨 Setting tokyo-night theme..."
  ln -sfn ~/.config/omarchy/themes/tokyo-night ~/.config/omarchy/current/theme
  if [ -f ~/.config/omarchy/current/theme/backgrounds.sh ]; then
    source ~/.config/omarchy/current/theme/backgrounds.sh
  fi
else
  # Find first available theme
  first_theme=$(find ~/.config/omarchy/themes -maxdepth 1 -type l -o -type d | grep -v '^\.$' | head -1)
  if [ -n "$first_theme" ]; then
    theme_name=$(basename "$first_theme")
    echo "🎨 Setting fallback theme: $theme_name..."
    ln -sfn "$first_theme" ~/.config/omarchy/current/theme
    if [ -f ~/.config/omarchy/current/theme/backgrounds.sh ]; then
      source ~/.config/omarchy/current/theme/backgrounds.sh 2>/dev/null || echo "⚠️  Could not source backgrounds for $theme_name"
    fi
  else
    echo "⚠️  No themes found, skipping theme setup"
    exit 0
  fi
fi

# Set background if available
if [ -d ~/.config/omarchy/backgrounds ]; then
  # Try to find the theme-specific background directory
  theme_name=$(basename "$(readlink ~/.config/omarchy/current/theme)" 2>/dev/null)
  if [ -n "$theme_name" ] && [ -d ~/.config/omarchy/backgrounds/"$theme_name" ]; then
    ln -sfn ~/.config/omarchy/backgrounds/"$theme_name" ~/.config/omarchy/current/backgrounds
    # Set first available background image as current
    first_bg=$(find ~/.config/omarchy/current/backgrounds -name "*.jpg" -o -name "*.png" | head -1)
    if [ -n "$first_bg" ]; then
      ln -sfn "$first_bg" ~/.config/omarchy/current/background
    fi
  fi
fi

# Set specific app links for current theme - only if theme exists and files are available
if [ -d ~/.config/omarchy/current/theme ]; then
  # Neovim theme
  if [ -f ~/.config/omarchy/current/theme/neovim.lua ]; then
    mkdir -p ~/.config/nvim/lua/plugins
    ln -sfn ~/.config/omarchy/current/theme/neovim.lua ~/.config/nvim/lua/plugins/theme.lua
  fi
  
  # btop theme
  if [ -f ~/.config/omarchy/current/theme/btop.theme ]; then
    mkdir -p ~/.config/btop/themes
    ln -sfn ~/.config/omarchy/current/theme/btop.theme ~/.config/btop/themes/current.theme
  fi
  
  # Mako configuration
  if [ -f ~/.config/omarchy/current/theme/mako.ini ]; then
    mkdir -p ~/.config/mako
    ln -sfn ~/.config/omarchy/current/theme/mako.ini ~/.config/mako/config
  fi
  
  echo "✅ Theme configuration completed"
else
  echo "⚠️  No theme directory found, app-specific themes not configured"
fi
