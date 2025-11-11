# test_gemini_only.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

def test_gemini_comprehensive():
    print("🔍 Comprehensive Gemini API Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("API_KEY_GEMINI")
    print(f"📋 API Key present: {bool(api_key)}")
    
    if not api_key:
        print("❌ ERROR: No API_KEY_GEMINI found in .env file")
        print("💡 Make sure your .env file has: API_KEY_GEMINI=your_actual_key_here")
        return
    
    if api_key == "YOUR_GEMINI_API_KEY":
        print("❌ ERROR: You're still using the placeholder API key")
        print("💡 Replace 'YOUR_GEMINI_API_KEY' with your actual key in the .env file")
        return
    
    print(f"✅ API Key loaded (first 10 chars): {api_key[:10]}...")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        print("✅ Gemini configured successfully")
        
        # List available models
        print("\n🔍 Checking available models...")
        models = genai.list_models()
        
        generation_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                generation_models.append(model.name)
                print(f"   ✅ {model.name}")
        
        if not generation_models:
            print("   ❌ No models support generateContent method")
            print("   💡 Available models (without generation):")
            for model in models:
                print(f"      - {model.name}")
            return
        
        print(f"\n🎯 Found {len(generation_models)} models that support content generation")
        
        # Try each available model
        for model_name in generation_models:
            print(f"\n🧪 Testing model: {model_name}")
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("What is 2+2? Answer in one word.")
                print(f"   ✅ SUCCESS: {response.text}")
                print(f"   💡 Use this model name in your code: '{model_name}'")
                break  # Stop at first successful model
            except Exception as e:
                print(f"   ❌ Failed with {model_name}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Configuration error: {e}")

if __name__ == "__main__":
    test_gemini_comprehensive()