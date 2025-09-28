#!/usr/bin/env python3
"""
Setup Git authentication for Raspberry Pi deployment
"""

import os
import subprocess
import sys

def main():
    print("🔧 Git Authentication Setup for Raspberry Pi")
    print("=" * 50)
    
    print("\n📋 Current Git Configuration:")
    try:
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, text=True, cwd='.')
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🚀 Quick Fix Options:")
    print("=" * 30)
    
    print("\n1️⃣ Option 1: Use Personal Access Token (Easiest)")
    print("   Steps:")
    print("   a) Go to: https://github.com/settings/tokens")
    print("   b) Click 'Generate new token (classic)'")
    print("   c) Select scopes: 'repo', 'workflow'")
    print("   d) Copy the token")
    print("   e) Run on Pi:")
    print("      git remote set-url origin https://github.com/wei12f8158/ExoGlove-YOLOv8.git")
    print("      git push origin master")
    print("      # Username: wei12f8158")
    print("      # Password: [paste your token]")
    
    print("\n2️⃣ Option 2: Use SSH Keys (Recommended)")
    print("   Steps:")
    print("   a) On Raspberry Pi:")
    print("      ssh-keygen -t ed25519 -C 'your-email@example.com'")
    print("      cat ~/.ssh/id_ed25519.pub")
    print("   b) Copy the public key")
    print("   c) Go to: https://github.com/settings/ssh/new")
    print("   d) Paste the key and save")
    print("   e) Test: ssh -T git@github.com")
    
    print("\n3️⃣ Option 3: Manual File Transfer")
    print("   If Git still doesn't work:")
    print("   a) Copy files from Mac to Pi using SCP:")
    print("      scp -r /path/to/ExoGlove pi@<pi-ip>:/home/pi/")
    print("   b) Or use USB drive")
    print("   c) Or use GitHub web interface")
    
    print("\n📚 GitHub Token Creation:")
    print("=" * 30)
    print("1. Visit: https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Note: 'Raspberry Pi ExoGlove Deployment'")
    print("4. Expiration: 90 days (or longer)")
    print("5. Select scopes:")
    print("   ✅ repo (Full control of private repositories)")
    print("   ✅ workflow (Update GitHub Action workflows)")
    print("6. Click 'Generate token'")
    print("7. Copy the token immediately (you won't see it again!)")
    
    print("\n🔑 Token Usage on Raspberry Pi:")
    print("=" * 35)
    print("git remote set-url origin https://github.com/wei12f8158/ExoGlove-YOLOv8.git")
    print("git push origin master")
    print("# When prompted:")
    print("# Username: wei12f8158")
    print("# Password: [paste your token here]")
    
    print("\n✅ After successful authentication:")
    print("• You can push/pull normally")
    print("• Token works for 90 days")
    print("• Create new token before expiration")
    
    print("\n🚨 Security Notes:")
    print("• Never share your token")
    print("• Store it securely")
    print("• Revoke if compromised")
    print("• Use SSH keys for better security")

if __name__ == "__main__":
    main()
