from paths import AUTH_TOKEN_PATH

# auth_token_path

def load_auth_token():
    with open(AUTH_TOKEN_PATH) as f:
        lines = f.readlines()

    return lines[0]