
[app]
title = Binance Scanner
package.name = binancescanner
package.domain = com.abdrahim
source.dir = .
source.include_exts = py,ttf,json,txt,md
version = 1.0


requirements = python3,hostpython3,kivy,requests,urllib3,certifi,idna,charset-normalizer,numpy,arabic-reshaper,python-bidi==0.4.2


orientation = portrait
fullscreen = 0
android.permissions = INTERNET,WAKE_LOCK
android.api = 34
android.minapi = 26
android.archs = arm64-v8a
android.allow_backup = True
android.accept_sdk_license = True
android.ndk = 28c
android.ndk_api = 26

p4a.branch = v2026.05.09

[buildozer]
log_level = 2
warn_on_root = 1
