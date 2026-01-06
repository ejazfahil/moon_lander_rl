#!/bin/bash
# GitHub Setup Script for Moon Lander RL Project
# Run this script from inside /Users/fahilejaz/Downloads/moon_lander_rl

echo "🔧 Moon Lander RL - GitHub Setup"
echo "================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ] || [ ! -f "test.py" ]; then
    echo "❌ Error: Please run this script from the moon_lander_rl directory"
    echo "   cd /Users/fahilejaz/Downloads/moon_lander_rl"
    echo "   ./github_setup.sh"
    exit 1
fi

# Initialize Git repository
echo "📦 Initializing Git repository..."
git init

# Configure Git (update with your info if needed)
echo "👤 Configuring Git..."
git config user.name "ejazfahil"
git config user.email "your-email@example.com"  # Update this!

# Add all files
echo "📝 Adding files to Git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Complete Moon Lander RL project with DQN implementation

- Implemented Deep Q-Network for LunarLander-v3  
- Training script with visual rendering and checkpointing
- Test script with video recording capabilities
- 500 episodes trained with 10 saved checkpoints
- Comprehensive README with setup instructions
- Jupyter notebook for interactive training"

# Set branch to main
echo "🌿 Setting default branch to main..."
git branch -M main

# Add GitHub remote
echo "🔗 Adding GitHub remote..."
git remote add origin https://github.com/ejazfahil/moon_lander_rl.git

echo ""
echo "✅ Git repository configured!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  IMPORTANT - Next Steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   • Go to: https://github.com/new"
echo "   • Repository name: moon_lander_rl"
echo "   • Make it Public or Private"
echo "   • DO NOT initialize with README, .gitignore, or license"
echo "   • Click 'Create repository'"
echo ""
echo "2. Then push your code:"
echo "   git push -u origin main"
echo ""
echo "Note: If the repository already exists, you can force push with:"
echo "   git push -u origin main --force"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
