import sys, requests

resp = requests.get(f"https://jokes.cloudburst.host/njapi/punchline?joke={sys.argv[1]}")
print ("response:", resp.status_code, resp.text)
sys.stdout.flush()

