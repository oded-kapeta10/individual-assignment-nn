import requests
import json

# REPLACE THIS with your actual Vercel URL
BASE_URL = "https://individual-assignment-nn.vercel.app"


def test_chat():
    url = f"{BASE_URL}/api/prompt"

    # The data we want to send
    payload = {
        "question": "I’m looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?"
    }

    print(f"Sending request to {url}...")

    try:
        response = requests.post(url, json=payload)

        # Check if call was successful
        if response.status_code == 200:
            # data = response.json()
            # print("\n--- SUCCESS! ---")
            # print(f"AI Response: {data.get('response')}")
            # print(f"\nContext Retrieved: {len(data.get('context', []))} chunks")
            data = response.json()

            # ADD THIS LINE to see the full raw JSON format:
            print(json.dumps(data, indent=2))

            print("\n--- SUCCESS! ---")
        else:
            print("\n--- ERROR ---")
            print(f"Status Code: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Failed to connect: {e}")


if __name__ == "__main__":
    test_chat()