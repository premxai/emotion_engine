from .base import *

DEBUG = False

ALLOWED_HOSTS = [
    ".onrender.com",
    "localhost",
    "127.0.0.1",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
] + MIDDLEWARE[1:]

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
