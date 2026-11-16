

import re
import ipaddress
from urllib.parse import urlparse


def extract_url_features(url):

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Check whether hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # Count digits
    digit_count = sum(char.isdigit() for char in url)

    host_digit_count = sum(
        char.isdigit() for char in hostname
    )

    # Suspicious words
    suspicious_words = [
        "login", "verify", "account", "update",
        "secure", "banking", "confirm", "password",
        "signin", "payment", "wallet"
    ]

    suspicious_count = sum(
        word in url.lower()
        for word in suspicious_words
    )

    # URL shortening services
    shorteners = [
        "bit.ly", "tinyurl.com", "t.co",
        "goo.gl", "is.gd", "ow.ly"
    ]

    has_shortener = int(
        any(shortener in hostname.lower()
            for shortener in shorteners)
    )

    # Number of subdomains
    domain_parts = hostname.split(".")

    if has_ip:
        subdomain_count = 0
    else:
        subdomain_count = max(0, len(domain_parts) - 2)

    # Check port
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    features = {

        "url_length": len(url),

        "hostname_length": len(hostname),

        "path_length": len(path),

        "query_length": len(query),

        "count_dots": url.count("."),

        "count_hyphens": url.count("-"),

        "count_at": url.count("@"),

        "count_question": url.count("?"),

        "count_ampersand": url.count("&"),

        "count_equal": url.count("="),

        "count_underscore": url.count("_"),

        "count_tilde": url.count("~"),

        "count_percent": url.count("%"),

        "count_slash": url.count("/"),

        "count_colon": url.count(":"),

        "count_comma": url.count(","),

        "count_semicolon": url.count(";"),

        "count_dollar": url.count("$"),

        "count_space": url.count(" "),

        "digit_count": digit_count,

        "digit_ratio": digit_count / max(len(url), 1),

        "host_digit_ratio": (
            host_digit_count / max(len(hostname), 1)
        ),

        "has_ip_address": has_ip,

        "has_https_token": int(
            "https" in hostname.lower()
            or "https" in path.lower()
        ),

        "has_www": int("www." in hostname.lower()),

        "www_count": url.lower().count("www"),

        "has_punycode": int("xn--" in hostname.lower()),

        "has_port": has_port,

        "has_prefix_suffix": int("-" in hostname),

        "subdomain_count": subdomain_count,

        "has_suspicious_word": int(suspicious_count > 0),

        "suspicious_word_count": suspicious_count,

        "has_shortener": has_shortener,

        "has_double_slash_in_path": int("//" in path),

        "path_has_extension": int(
            bool(re.search(r"\.[a-zA-Z0-9]{1,6}$", path))
        ),

        "domain_length": len(hostname)
    }

    return features
