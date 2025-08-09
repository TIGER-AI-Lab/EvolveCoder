#!/usr/bin/env python3
"""
🔑 OpenAI API Key Validation Tool

This tool helps diagnose and validate OpenAI API key issues.
"""

import os
import sys
import openai
from pathlib import Path

def test_api_key(api_key: str) -> bool:
    """Test if an API key is valid by making a simple API call."""
    
    if not api_key or not api_key.strip():
        print("❌ API key is empty")
        return False
    
    api_key = api_key.strip()
    
    print(f"🔍 Testing API key: {api_key[:20]}...")
    
    try:
        # Create client
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple test call
        print("📞 Making test API call...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ API key is valid!")
        print(f"📊 Response: {response.choices[0].message.content}")
        return True
        
    except openai.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        return False
    except openai.RateLimitError as e:
        print(f"⚠️ Rate limit exceeded (but key is valid): {e}")
        return True  # Key is valid, just rate limited
    except openai.PermissionDeniedError as e:
        print(f"⚠️ Permission denied (key valid but no access to model): {e}")
        return True  # Key is valid, just no permission for this model
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_api_key_format(api_key: str) -> bool:
    """Check if API key has the correct format."""
    
    if not api_key:
        print("❌ API key is empty")
        return False
    
    api_key = api_key.strip()
    
    print(f"🔍 Checking API key format...")
    print(f"   Length: {len(api_key)} characters")
    print(f"   Starts with: {api_key[:10]}...")
    print(f"   Ends with: ...{api_key[-10:]}")
    
    # Check format
    if not api_key.startswith('sk-'):
        print("❌ API key should start with 'sk-'")
        return False
    
    if len(api_key) < 20:
        print("❌ API key seems too short")
        return False
    
    print("✅ API key format looks correct")
    return True

def get_new_api_key_instructions():
    """Provide instructions for getting a new API key."""
    
    print("\n🔑 HOW TO GET A NEW OPENAI API KEY")
    print("=" * 50)
    print("1. 🌐 Go to: https://platform.openai.com/account/api-keys")
    print("2. 🔐 Log in to your OpenAI account")
    print("3. ➕ Click 'Create new secret key'")
    print("4. 📝 Give it a name (e.g., 'AceCoder Project')")
    print("5. 📋 Copy the key immediately (you won't see it again!)")
    print("6. 💰 Make sure you have credit balance in your account")
    print("7. ⚙️ Replace the key in your environment or interface")
    
    print("\n💡 COMMON ISSUES:")
    print("   • Key expired or was revoked")
    print("   • No credit balance in OpenAI account")
    print("   • Key doesn't have access to the model (gpt-4.1-mini)")
    print("   • Typo when copying/pasting the key")

def interactive_key_test():
    """Interactive API key testing."""
    
    print("🔑 INTERACTIVE API KEY VALIDATION")
    print("=" * 50)
    
    # Check environment key first
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key:
        print(f"\n📋 Found key in environment: {env_key[:20]}...")
        if check_api_key_format(env_key):
            if test_api_key(env_key):
                print("🎉 Environment API key is working!")
                return True
            else:
                print("❌ Environment API key is not working")
        else:
            print("❌ Environment API key format is invalid")
    else:
        print("⚠️ No API key found in environment")
    
    # Ask user to input a key to test
    print("\n🔄 Let's test a new API key...")
    while True:
        try:
            new_key = input("\n🔑 Enter your OpenAI API key (or 'quit' to exit): ").strip()
            
            if new_key.lower() == 'quit':
                break
            
            if not new_key:
                print("❌ Please enter a valid API key")
                continue
            
            if check_api_key_format(new_key):
                if test_api_key(new_key):
                    print("\n🎉 SUCCESS! This API key works!")
                    
                    # Offer to update environment
                    update = input("\n💾 Update environment variable? (y/n): ").lower().strip()
                    if update == 'y':
                        os.environ['OPENAI_API_KEY'] = new_key
                        print("✅ Environment variable updated for this session")
                        print("💡 To persist, add to your shell profile:")
                        print(f"   export OPENAI_API_KEY='{new_key}'")
                    
                    return True
                else:
                    print("❌ This API key doesn't work")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def main():
    """Main validation function."""
    
    print("🔑 OpenAI API Key Validation Tool")
    print("=" * 40)
    
    # Test current key
    success = interactive_key_test()
    
    if not success:
        get_new_api_key_instructions()
        print("\n🔄 Once you have a new API key, run this tool again to test it")
        return 1
    
    print("\n🎯 NEXT STEPS:")
    print("   1. ✅ Your API key is working")
    print("   2. 🚀 You can now run the adversarial generation pipeline")
    print("   3. 🎨 Use the Gradio interface: python app.py")
    print("   4. 💡 Or run directly: python main.py --rounds 1")
    
    return 0

if __name__ == "__main__":
    exit(main())
