import os
import json
import urllib.request
import urllib.error

# Read the API key from the environment
api_key = os.getenv("OPENAQ_API_KEY")

if not api_key:
    print("API key not found.")
    print("Please set the key again in your terminal.")
else:
    url = "https://api.openaq.org/v3/locations/8118"

    request = urllib.request.Request(
        url,
        headers={"X-API-Key": api_key}
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        location = data["results"][0]

        print("AirAware connected successfully!")
        print("Location:", location["name"])
        print("Country:", location["country"]["name"])
        print("API connection is working.")

    except urllib.error.HTTPError as error:
        print("API returned an error:", error.code)
        print("Check your API key and account access.")

    except Exception as error:
        print("Connection failed:", error)