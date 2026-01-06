#!/bin/bash
# Push Moon Lander RL to GitHub
# Repository: https://github.com/ejazfahil/moon_lander_rl

echo "🚀 Pushing Moon Lander RL to GitHub"
echo "===================================="
echo ""

# Navigate to project directory
cd /Users/fahilejaz/Downloads/moon_lander_rl || exit 1

# Initialize Git if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git branch -M main
else
    echo "✓ Git repository already initialized"
fi

# Configure Git user (update email if needed)
echo "👤 Configuring Git user..."
git config user.name "ejazfahil" 2>/dev/null || true
git config user.email "ejazfahil@users.noreply.github.com" 2>/dev/null || true

# Add all files
echo "📝 Adding files to Git..."
git add .

# Create commit
echo "💾 Creating commit..."
git commit -m "Complete Moon Lander RL project with DQN implementation

- Deep Q-Network implementation for LunarLander-v3
- Training script with real-time visual rendering
- Test script with video recording capabilities
- 500 episodes trained with 10 saved checkpoints
- Comprehensive README and documentation
- Jupyter notebook for interactive training
- Automated deployment scripts" || echo "Nothing to commit or already committed"

# Add remote (if not already added)
echo "🔗 Adding GitHub remote..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/ejazfahil/moon_lander_rl.git

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Successfully pushed to GitHub!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 View your repository at:"
echo "   https://github.com/ejazfahil/moon_lander_rl"
echo ""
