# -*- coding: utf-8 -*-
"""
Binance Scanner — تطبيق أندرويد (Kivy)

- الاستراتيجيات الأساسية: scanner_v5 و scanner_htf
- أي ملف strategies/strategy_*.py بيظهر لوحده كـ كارت جديد (شوف README)
"""

import importlib
import json
import os
import queue
import re
import sys
import textwrap
import unicodedata
import threading
import time as _time

try:  # اختياريان: بيُستخدما فقط في وضع "تشكيل يدوي"
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:  # pragma: no cover
    arabic_reshaper = None
    get_display = None

from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import binance_utils  # noqa: E402  (لازم بعد ضبط sys.path)

# ---------------- الألوان ----------------
C_BG = (13 / 255, 17 / 255, 23 / 255, 1)
C_CARD = (22 / 255, 27 / 255, 34 / 255, 1)
C_GREEN = (22 / 255, 199 / 255, 132 / 255, 1)
C_RED = (234 / 255, 57 / 255, 67 / 255, 1)
C_TEXT = (230 / 255, 237 / 255, 243 / 255, 1)
C_MUTED = (139 / 255, 148 / 255, 158 / 255, 1)
C_YELLOW = (240 / 255, 185 / 255, 41 / 255, 1)

FONT = "Roboto"
FONT_BOLD = "Roboto"

# ---------------- الإعدادات (تُحفظ في ملف) ----------------
# ar_mode:
#   raw    : نص عربي خام — Kivy على أغلب الأجهزة بيشكّله ويظبط اتجاهه بنفسه (الافتراضي)
#   shaped : تشكيل + اتجاه يدوي (arabic_reshaper + bidi) — لو الخام طلع حروف مقطعة
#   rlm    : نص خام مع علامة اتجاه يمين->يسار في أول كل سطر
AR_MODES = ["raw", "shaped", "rlm"]
AR_MODE_NAMES = {
    "raw": "خام (الافتراضي)",
    "shaped": "تشكيل يدوي",
    "rlm": "خام + علامة اتجاه",
}
CFG = {"ar_mode": "raw", "hide_rejected": True}
_CFG_PATH = None
PREVIEW = "فلتر 4h: مطلوب (4h صاعد) | الدخول 0.9996\nانتهى في 133 ثانية"


def load_cfg(path):
    global _CFG_PATH
    _CFG_PATH = path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("ar_mode") in AR_MODES:
            CFG["ar_mode"] = data["ar_mode"]
        CFG["hide_rejected"] = bool(data.get("hide_rejected", True))
    except Exception:
        pass


def save_cfg():
    if not _CFG_PATH:
        return
    try:
        with open(_CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(CFG, f)
    except OSError:
        pass


def setup_font():
    global FONT, FONT_BOLD
    for rel, is_bold in (("fonts/Tajawal-Regular.ttf", False), ("Tajawal-Regular.ttf", False),
                         ("fonts/Tajawal-Bold.ttf", True), ("Tajawal-Bold.ttf", True)):
        path = os.path.join(BASE_DIR, rel)
        if os.path.exists(path):
            if is_bold:
                FONT_BOLD = path
            else:
                FONT = path
    if FONT_BOLD == "Roboto" and FONT != "Roboto":
        FONT_BOLD = FONT


# ---------------- معالجة النصوص ----------------
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_EMOJI_MAP = {
    "🟢": "•", "🔴": "•", "⚪": "·", "🔎": "»", "🔍": "»", "📈": "+", "📊": "#",
    "⚠️": "!", "⚠": "!", "✅": "+", "💾": "-", "ℹ️": "i", "ℹ": "i", "⏹": "-", "🚀": "»",
}
# خط Tajawal ناقص كل الأشكال "المنفردة" للحروف العربية (مثل الألف في أول الكلمة)،
# فبنرجعها للحرف العادي (نفس الشكل) في وضع "تشكيل يدوي".
_ISOLATED = [0xfe80, 0xfe81, 0xfe83, 0xfe85, 0xfe87, 0xfe89, 0xfe8d, 0xfe8f, 0xfe93, 0xfe95,
             0xfe99, 0xfe9d, 0xfea1, 0xfea5, 0xfea9, 0xfeab, 0xfead, 0xfeaf, 0xfeb1, 0xfeb5,
             0xfeb9, 0xfebd, 0xfec1, 0xfec5, 0xfec9, 0xfecd, 0xfed1, 0xfed5, 0xfed9, 0xfedd,
             0xfee1, 0xfee5, 0xfee9, 0xfeed, 0xfeef, 0xfef1]
_ISO_FIX = {c: unicodedata.normalize("NFKC", chr(c)) for c in _ISOLATED}
_STRIP_RE = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F\u200d]")


def has_arabic(text):
    return bool(_AR_RE.search(text))


def clean(text):
    """يشيل الإيموجي (الخط مش بيدعمها فبتطلع مربعات)."""
    for k, v in _EMOJI_MAP.items():
        text = text.replace(k, v)
    return _STRIP_RE.sub("", text)


def ar(text):
    """يجهز نص للعرض حسب وضع العربي المختار."""
    text = str(text)
    mode = CFG["ar_mode"]
    if mode == "shaped" and arabic_reshaper and get_display and has_arabic(text):
        try:
            return get_display(arabic_reshaper.reshape(text)).translate(_ISO_FIX)
        except Exception:
            return text
    if mode == "rlm" and has_arabic(text):
        return "\u200f" + text
    return text


def ar_block(text, chars_per_line=40):
    """للنصوص متعددة الأسطر: في وضع 'تشكيل يدوي' بنقسم الأسطر قبل الـ bidi."""
    if CFG["ar_mode"] != "shaped":
        return "\n".join(ar(line) for line in str(text).split("\n"))
    out = []
    for line in str(text).split("\n"):
        for chunk in (textwrap.wrap(line, chars_per_line) or [""]):
            out.append(ar(chunk))
    return "\n".join(out)


# ---------------- سجل الاستراتيجيات ----------------
BUILTIN = [
    {
        "module": "scanner_v5",
        "name": "فحص بدون فلتر صاعد",
        "desc": "موديل V5 — توازن تكيفي + Order Flow حقيقي\nتأكيد 15m | إشارة من 85 نقطة",
    },
    {
        "module": "scanner_htf",
        "name": "فحص بفلتر صاعد",
        "desc": "فحص مع فلتر اتجاه 4h إلزامي\n(EMA50 + ميل صاعد) | إشارة من 13 نقطة",
    },
]


def load_registry():
    """الأساسية + أي strategies/strategy_*.py. عطل في استراتيجية ما يوقفش الباقي."""
    registry = [dict(item) for item in BUILTIN]
    sdir = os.path.join(BASE_DIR, "strategies")
    names = set()
    if os.path.isdir(sdir):
        for fname in os.listdir(sdir):
            m = re.match(r"(strategy_\w+)\.(py|pyc)$", fname)
            if m:
                names.add(m.group(1))
    for name in sorted(names):
        modname = f"strategies.{name}"
        try:
            mod = importlib.import_module(modname)
            if not callable(getattr(mod, "scan_all", None)):
                raise AttributeError("الملف ناقصه دالة scan_all()")
            registry.append({"module": modname,
                             "name": str(getattr(mod, "NAME", name)),
                             "desc": str(getattr(mod, "DESC", ""))})
        except Exception as exc:
            registry.append({"module": modname, "name": name,
                             "desc": f"خطأ في التحميل: {type(exc).__name__}: {exc}",
                             "error": True})
    return registry


# ---------------- الإيقاف وتحويل print ----------------
StopScan = binance_utils.StopScan


class QueueWriter:
    """يحوّل مخرجات print داخل السكربت إلى صف رسائل يعرضه التطبيق."""

    def __init__(self, out_queue):
        self.out_queue = out_queue
        self.buf = ""

    def write(self, text):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.out_queue.put(line.rstrip())

    def flush(self):
        if self.buf.strip():
            self.out_queue.put(self.buf.rstrip())
        self.buf = ""


class StopAwareTime:
    """بديل time للسكربتات القديمة: sleep يستجيب للإيقاف فوراً."""

    def sleep(self, seconds):
        binance_utils.sleep(float(seconds))

    def time(self):
        return _time.time()

    def monotonic(self):
        return _time.monotonic()


# ---------------- عناصر الواجهة ----------------
def make_button(text, bg, height=48, size=16, **kw):
    return Button(text=ar(text), color=C_TEXT, background_color=bg, background_normal="",
                  background_down="", size_hint_y=None, height=dp(height),
                  font_size=sp(size), font_name=FONT, **kw)


def make_label(text, color=C_TEXT, size=14, bold=False, height=None, halign="center", **kw):
    lbl = Label(text=ar_block(text), color=color, font_size=sp(size),
                font_name=FONT_BOLD if bold else FONT, halign=halign, valign="middle", **kw)
    if height is not None:
        lbl.size_hint_y = None
        lbl.height = dp(height)
    lbl.bind(size=lambda inst, s: setattr(inst, "text_size", (s[0], None)))
    return lbl


class Card(ButtonBehavior, BoxLayout):
    def __init__(self, title, desc, error=False, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(16)
        self.spacing = dp(6)
        self.size_hint_y = None
        self.height = dp(120)
        with self.canvas.before:
            Color(*C_CARD)
            self._bg = RoundedRectangle(radius=[dp(16)], pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)
        self.add_widget(make_label(title, C_RED if error else C_TEXT, 19, bold=True, height=30))
        self.add_widget(make_label(desc, C_MUTED, 13))

    def _upd(self, *a):
        self._bg.pos, self._bg.size = self.pos, self.size

    def on_press(self):
        self._bg.radius = [dp(10)]

    def on_release(self):
        self._bg.radius = [dp(16)]


class HomeScreen(Screen):
    def __init__(self, registry, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))
        root.add_widget(make_label("Binance Scanner", C_TEXT, 28, bold=True, height=44))
        root.add_widget(make_label("فاحص سوق باينانس — إشارات شراء سبوت فقط", C_MUTED, 13, height=22))

        scroll = ScrollView(do_scroll_x=False)
        grid = GridLayout(cols=1, spacing=dp(12), size_hint_y=None, padding=(0, dp(8)))
        grid.bind(minimum_height=grid.setter("height"))
        for item in registry:
            card = Card(item["name"], item["desc"], error=item.get("error", False))
            if not item.get("error"):
                card.bind(on_release=lambda c, m=item["module"], n=item["name"]:
                          App.get_running_app().open_scan(m, n))
            grid.add_widget(card)
        scroll.add_widget(grid)
        root.add_widget(scroll)

        settings_btn = make_button("الإعدادات", C_CARD, 50)
        settings_btn.bind(on_press=lambda b: self.show_settings())
        root.add_widget(settings_btn)
        root.add_widget(make_label("إشارات تحليلية فقط — لا يوجد تنفيذ أوامر", C_MUTED, 11, height=20))
        self.add_widget(root)

    def show_settings(self):
        box = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(10))
        preview = make_label("", C_YELLOW, 13)
        mode_btn = make_button("", C_CARD, 46, 15)
        hide_btn = make_button("", C_CARD, 46, 15)

        def refresh():
            mode_btn.text = ar(f"وضع العربي: {AR_MODE_NAMES[CFG['ar_mode']]}")
            hide_btn.text = ar("إخفاء المرفوضات: " + ("نعم" if CFG["hide_rejected"] else "لا"))
            preview.text = ar_block("معاينة: " + PREVIEW, 34)

        def cycle_mode(_):
            CFG["ar_mode"] = AR_MODES[(AR_MODES.index(CFG["ar_mode"]) + 1) % len(AR_MODES)]
            save_cfg()
            refresh()

        def toggle_hide(_):
            CFG["hide_rejected"] = not CFG["hide_rejected"]
            save_cfg()
            refresh()

        mode_btn.bind(on_press=cycle_mode)
        hide_btn.bind(on_press=toggle_hide)
        hint = make_label("لو العربي ظاهر معكوس أو بحروف مقطعة، اضغط 'وضع العربي' "
                          "لحد ما المعاينة تبقى صح.", C_MUTED, 12)
        close_btn = make_button("إغلاق", C_GREEN, 46, 15)
        for w in (preview, mode_btn, hide_btn, hint, close_btn):
            box.add_widget(w)
        refresh()
        popup = Popup(title=ar("الإعدادات"), title_font=FONT, content=box, size_hint=(0.92, 0.62),
                      background_color=C_BG, separator_color=C_GREEN)
        close_btn.bind(on_press=lambda b: (popup.dismiss(), App.get_running_app().rebuild()))
        popup.open()


class ConsoleView(ScrollView):
    """كونسول: كل سطر Label مستقل (آمن على حد الـ texture في أندرويد)."""

    MAX_WIDGETS = 400
    MAX_KEPT = 20000

    def __init__(self, **kwargs):
        super().__init__(do_scroll_x=False, bar_width=dp(4), **kwargs)
        self.grid = GridLayout(cols=1, spacing=dp(2), size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter("height"))
        self.add_widget(self.grid)
        self.kept = []

    def reset(self):
        self.grid.clear_widgets()
        self.kept = []

    @staticmethod
    def classify(line):
        s = line.strip()
        low = s.lower()
        if s.startswith(("🟢 BUY", "🟢 شراء", "• BUY", "• شراء")):
            return "buy"
        if s.upper().startswith("REJECT") or "rejected (see" in low or "تخطي" in s or "skipped" in low:
            return "muted"
        if "⚠" in s or "error" in low or "خطأ" in s:
            return "warn"
        return "normal"

    def add_lines(self, lines):
        stick = self.scroll_y <= 0.05
        for raw in lines:
            self.kept.append(raw)
            kind = self.classify(raw)
            if kind == "muted" and CFG["hide_rejected"]:
                continue
            text = clean(raw)
            color = {"buy": C_GREEN, "muted": C_MUTED, "warn": C_YELLOW}.get(kind, C_TEXT)
            rtl = has_arabic(text)
            if CFG["ar_mode"] == "shaped" and rtl:
                width_chars = max(20, int(Window.width / (sp(12) * 0.55)))
                shown = ar_block(text, width_chars)
            else:
                shown = ar(text)
            lbl = Label(text=shown, color=color, font_size=sp(12), font_name=FONT,
                        halign="right" if rtl else "left", valign="top",
                        size_hint_y=None, markup=False)
            lbl.bind(width=lambda i, w: setattr(i, "text_size", (w, None)))
            lbl.bind(texture_size=lambda i, s: setattr(i, "height", s[1] + dp(2)))
            self.grid.add_widget(lbl)
        while len(self.grid.children) > self.MAX_WIDGETS:
            self.grid.remove_widget(self.grid.children[-1])
        if len(self.kept) > self.MAX_KEPT:
            del self.kept[: len(self.kept) - self.MAX_KEPT]
        if stick:
            Clock.schedule_once(lambda dt: setattr(self, "scroll_y", 0), 0.05)

    def full_text(self):
        return "\n".join(clean(x) for x in self.kept)


class ScanScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        header = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(70), spacing=dp(2))
        self.title_label = make_label("", C_TEXT, 19, bold=True, height=32)
        self.status_label = make_label("", C_GREEN, 13, height=24)
        header.add_widget(self.title_label)
        header.add_widget(self.status_label)
        layout.add_widget(header)

        self.console = ConsoleView()
        layout.add_widget(self.console)

        self.stop_btn = make_button("إيقاف الفحص", C_RED, 52, 17)
        self.stop_btn.bind(on_press=lambda b: self.request_stop())
        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.copy_btn = make_button("نسخ النتائج", C_CARD, 44, 14)
        self.copy_btn.bind(on_press=lambda b: self.copy_results())
        self.back_btn = make_button("رجوع", C_CARD, 44, 14)
        self.back_btn.bind(on_press=lambda b: self.go_back())
        row.add_widget(self.copy_btn)
        row.add_widget(self.back_btn)
        layout.add_widget(self.stop_btn)
        layout.add_widget(row)
        self.add_widget(layout)

        self.out_queue = queue.Queue()
        self.worker = None
        self.started_at = None
        self.signals = 0
        self.stop_requested = False

    def open_scan(self, module_name, name):
        self.title_label.text = ar(name)
        self.status_label.color = C_GREEN
        self.status_label.text = ar("جارٍ التشغيل...")
        self.console.reset()
        self.out_queue = queue.Queue()
        self.signals = 0
        self.stop_requested = False
        binance_utils.STOP.clear()
        self.stop_btn.disabled = False
        self.back_btn.disabled = True
        self.started_at = _time.time()
        self.worker = threading.Thread(target=self._run_scan, args=(module_name,), daemon=True)
        self.worker.start()
        Clock.unschedule(self._drain)
        Clock.unschedule(self._tick)
        Clock.schedule_interval(self._drain, 0.4)
        Clock.schedule_interval(self._tick, 1.0)

    def _run_scan(self, module_name):
        old_stdout = sys.stdout
        sys.stdout = QueueWriter(self.out_queue)
        try:
            try:
                os.chdir(App.get_running_app().user_data_dir)  # مكان قابل للكتابة لملفات السجل
            except Exception:
                pass
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            module.time = StopAwareTime()          # للسكربتات القديمة اللي بتستخدم time.sleep
            if CFG["hide_rejected"] and hasattr(module, "PRINT_REJECTED"):
                module.PRINT_REJECTED = False      # يقلل الزحمة ويسرّع الشاشة
            module.scan_all()
        except StopScan:
            self.out_queue.put("- تم إيقاف الفحص بواسطة المستخدم.")
        except ImportError as exc:
            self.out_queue.put(f"خطأ في تحميل الوحدة: {exc}")
        except BaseException as exc:
            self.out_queue.put(f"⚠️ حدث خطأ: {type(exc).__name__}: {exc}")
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            sys.stdout = old_stdout
            self.out_queue.put("__DONE__")

    def _drain(self, dt):
        lines, done = [], False
        for _ in range(500):  # حد لكل دورة عشان الواجهة ما تهنجش
            try:
                item = self.out_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__DONE__":
                done = True
            else:
                lines.append(item)
        if lines:
            self.signals += sum(1 for l in lines if self.console.classify(l) == "buy")
            self.console.add_lines(lines)
        if done:
            self._finish()

    def _tick(self, dt):
        if self.started_at and not self.stop_requested:
            m, s = divmod(int(_time.time() - self.started_at), 60)
            self.status_label.text = ar(f"يعمل الآن {m:02d}:{s:02d} | إشارات: {self.signals}")

    def request_stop(self):
        self.stop_requested = True
        binance_utils.STOP.set()
        self.status_label.text = ar("جارٍ الإيقاف...")
        self.status_label.color = C_YELLOW
        self.stop_btn.disabled = True

    def _finish(self):
        Clock.unschedule(self._tick)
        Clock.unschedule(self._drain)
        self.stop_btn.disabled = True
        self.back_btn.disabled = False
        elapsed = int(_time.time() - self.started_at)
        self.status_label.color = C_GREEN
        self.status_label.text = ar(f"انتهى في {elapsed} ثانية | إشارات: {self.signals}")

    def copy_results(self):
        try:
            Clipboard.copy(self.console.full_text())
            self.status_label.text = ar("تم نسخ النتائج")
        except Exception as exc:
            self.status_label.text = ar(f"تعذر النسخ: {exc}")

    def go_back(self):
        if self.worker and self.worker.is_alive():
            return
        App.get_running_app().go_home()


class ScannerApp(App):
    def build(self):
        setup_font()
        load_cfg(os.path.join(self.user_data_dir, "settings.json"))
        Window.clearcolor = C_BG
        try:
            Window.allow_screensaver = False   # الشاشة ما تنطفيش أثناء الفحص
        except Exception:
            pass
        self.sm = ScreenManager()
        self._populate()
        return self.sm

    def _populate(self):
        self.registry = load_registry()
        self.home = HomeScreen(self.registry, name="home")
        self.scan = ScanScreen(name="scan")
        self.sm.add_widget(self.home)
        self.sm.add_widget(self.scan)
        self.sm.current = "home"

    def rebuild(self):
        """إعادة بناء الواجهة (بعد تغيير وضع العربي)."""
        if self.scan.worker and self.scan.worker.is_alive():
            return
        self.sm.clear_widgets()
        self._populate()

    def open_scan(self, module_name, name):
        self.sm.current = "scan"
        self.scan.open_scan(module_name, name)

    def go_home(self):
        self.sm.current = "home"


if __name__ == "__main__":
    ScannerApp().run()
