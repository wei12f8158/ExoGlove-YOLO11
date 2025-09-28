# Fix Git Authentication for Raspberry Pi

## Problem
GitHub no longer accepts password authentication. You need to use either:
1. SSH keys (recommended)
2. Personal Access Token (PAT)

## Solution 1: SSH Keys (Recommended)

### On Raspberry Pi:
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to SSH agent
ssh-add ~/.ssh/id_ed25519

# Display public key
cat ~/.ssh/id_ed25519.pub
```

### On GitHub:
1. Go to GitHub.com → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste the public key from above
4. Give it a title like "Raspberry Pi ExoGlove"

### Test SSH connection:
```bash
ssh -T git@github.com
```

## Solution 2: Personal Access Token

### Create PAT on GitHub:
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`, `workflow`
4. Copy the token (save it securely!)

### On Raspberry Pi:
```bash
# Change remote to HTTPS with token
git remote set-url origin https://github.com/wei12f8158/ExoGlove-YOLOv8.git

# Use token as password when prompted
git push origin master
# Username: wei12f8158
# Password: [paste your PAT token]
```

## Solution 3: Quick Fix - Use SSH

```bash
# Make sure remote is SSH
git remote set-url origin git@github.com:wei12f8158/ExoGlove-YOLOv8.git

# Test SSH connection
ssh -T git@github.com

# If SSH works, push normally
git push origin master
```

## Troubleshooting

### If SSH key doesn't work:
```bash
# Check SSH key
ssh-add -l

# Add key again
ssh-add ~/.ssh/id_ed25519

# Test GitHub connection
ssh -T git@github.com
```

### If you get "Permission denied":
1. Make sure SSH key is added to GitHub
2. Check key permissions: `chmod 600 ~/.ssh/id_ed25519`
3. Try: `ssh -vT git@github.com` for detailed output
