import sys
import os
import sys
import os
import json # jsonを追加
import fitz  # PyMuPDF
import google.generativeai as genai
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QScrollArea, QSizePolicy, QFrame, QTextEdit, QDialog,
    QMessageBox, QTabWidget, QLineEdit, QComboBox, QFormLayout, QSpacerItem,
    QProgressDialog, QListWidget, QDialogButtonBox, QListWidgetItem, QToolBar, # QToolBar を追加
    QInputDialog # QInputDialog を追加 (ショートカット用)
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QAction, QPalette, QPainter, QFontMetrics, QTextCursor, QKeySequence # QKeySequence を追加
from PyQt6.QtCore import Qt, QSize, QTimer, QSettings, QThread, QObject, pyqtSignal

# --- 定数 ---
# 設定キー
SETTINGS_API_KEY = "gemini/apiKey"
SETTINGS_MODEL = "gemini/model"
SETTINGS_LAST_DIR = "general/lastDirectory" # 最後に開いたディレクトリ
SETTINGS_CUSTOM_PROMPTS = "general/customPrompts" # カスタムプロンプト用のキーを追加

# モデル
DEFAULT_MODEL = 'gemini-1.5-pro-latest' # デフォルトを安定版Proに変更
# ユーザーフィードバックに基づくAVAILABLE_MODELSの更新
AVAILABLE_MODELS = [
    'gemini-2.5-pro-exp-03-25', # 実験的 2.5 Pro (必要に応じてコメント解除)
    'gemini-2.0-flash',        # 次世代Flash（注意して使用）(必要に応じてコメント解除)
    'gemini-1.5-pro-latest',    # 安定版 Pro（推奨デフォルト）
    'gemini-1.5-flash-latest',  # 安定版 Flash
    'gemini-pro',               # レガシー Pro（互換性が必要な場合）
    'gemini-2.0-flash-lite',   # Lite Flash (必要に応じてコメント解除)
    'gemini-1.5-flash-8b',     # Small Flash (必要に応じてコメント解除)
]
# ユーザーが必要とする場合は実験的/特定のモデルを追加するが、デフォルトは安定させる

# UI定数
DEFAULT_WINDOW_WIDTH = 1000
DEFAULT_WINDOW_HEIGHT = 800
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0 # 最大ズーム制限を追加
ZOOM_INCREMENT = 1.2
ICON_SIZE = QSize(24, 24)
RESIZE_DELAY_MS = 100 # リサイズイベント処理の遅延
PROGRESS_DIALOG_DELAY_MS = 400 # プログレスダイアログ表示までの最小時間
STATUS_BAR_MSG_DURATION_MS = 3000 # ステータスバーメッセージの表示時間 (ミリ秒)
TWO_PAGE_SPACING_BASE = 10 # 見開き表示時のページ間基本スペーシング
FIT_PADDING = 15 # 幅/高さ合わせ計算時のパディング

# プロンプト
TRANSLATION_PROMPT_TEMPLATE = "以下のテキストを日本語に翻訳してください:\n\n---\n{text}\n---"
SUMMARIZE_PROMPT_TEMPLATE = "以下のテキストを日本語で簡潔に要約してください::\n\n---\n{text}\n---"
EXAMPLE_PROMPT_TEMPLATE = "以下のテキストの内容を説明する具体的な例を挙げてください。結果は日本語で出力してください:\n\n---\n{text}\n---"
EXPLAIN_TERM_PROMPT_TEMPLATE = "以下のテキストに含まれる専門用語や重要な概念をいくつか選び、それぞれを初心者にもわかるように日本語で説明してください:\n\n---\n{text}\n---"
FREE_TRANSLATION_PROMPT_TEMPLATE = "以下のテキストを、文の構成を変えてもよいので、非常に分かりやすい自然な日本語に意訳してください:\n\n---\n{text}\n---" # 意訳用プロンプトを追加
RECONSTRUCT_PROMPT_TEMPLATE = "以下のテキストの内容で伝えたいことを、別の言葉や具体的な例を用いて、より分かりやすく日本語で再構築してください:\n\n---\n{text}\n---" # 再構築用プロンプトを追加
INTERPRET_PROMPT_TEMPLATE = "以下のテキストが伝えようとしている主な意図や目的を解釈し、日本語で説明してください:\n\n---\n{text}\n---" # 解釈用プロンプトを追加
CUSTOM_PROMPT_PLACEHOLDER = "{text}" # プロンプト内のテキスト挿入箇所を示すプレースホルダー

# --- ヘルパー関数 ---
def get_icon(standard_name, fallback_path=None):
    """アイコンを取得します。フォールバックパスが存在すれば優先し、なければテーマを使用します。"""
    icon = QIcon() # 空のアイコンから開始
    if fallback_path and os.path.exists(fallback_path):
        icon = QIcon(fallback_path)
        if icon.isNull(): # ファイルからの読み込みに失敗したか確認
             print(f"警告: フォールバックパスからのアイコン読み込みに失敗しました: {fallback_path}")
             # ファイルが存在するが読み込みに失敗した場合でも、最後の手段としてテーマを試すことも可能
             # icon = QIcon.fromTheme(standard_name)
    else:
        # フォールバックパスが存在しないか提供されなかった場合、テーマを試す
        icon = QIcon.fromTheme(standard_name)

    if icon.isNull():
        # フォールバック（試行した場合）とテーマの両方が失敗した場合
        print(f"警告: パス '{fallback_path}' またはテーマ '{standard_name}' からアイコンが見つかりません。ボタンがアイコンなしで表示される可能性があります。")
    return icon

# --- バックグラウンドでのGemini API呼び出し用ワーカクラス ---
class GeminiWorker(QObject):
    """Gemini API呼び出しを別スレッドで実行します。"""
    finished = pyqtSignal(str) # 成功時に結果テキストを運ぶ
    error = pyqtSignal(str)    # 失敗時にエラーメッセージを運ぶ
    progress = pyqtSignal(int) # オプション: より詳細な進捗報告用

    def __init__(self, model, prompt):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self._is_interrupted = False

    def run(self):
        """API呼び出しを実行します。"""
        try:
            # Geminiモデルが設定されていない場合
            if not self.model:
                self.error.emit("Geminiモデルが設定されていません。")
                return

            # API呼び出し開始前に中断された場合
            if self._is_interrupted:
                self.error.emit("APIコール開始前にキャンセルされました。") # 一般的なメッセージ
                return

            # --- 実際のAPI呼び出し ---
            # 注意: generate_contentはブロッキングです。呼び出し中のリアルタイムな進捗更新は、
            # ストリーミングAPIが利用可能であれば、それが必要になる可能性があります。
            self.progress.emit(50) # 呼び出しが開始されたことを示す（任意の価）
            response = self.model.generate_content(self.prompt)
            # -----------------------

            # ブロッキング呼び出しが返された *後* に中断をチェック
            if self._is_interrupted:
                 print("API呼び出しが中断されました（API呼び出しが返された後）")
                 self.error.emit("APIコールがキャンセルされました。") # 一般的なメッセージ
                 return

            # 応答の有効性をチェック（基本的なチェック）
            if not hasattr(response, 'text') or not response.text:
                 # 可能であれば詳細情報を取得しようとする（APIライブラリに依存）
                 error_detail = "不明な応答形式"
                 try:
                     # ライブラリが提供している場合、特定のエラー属性を確認する
                     if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                         error_detail = f"プロンプトフィードバック: {response.prompt_feedback}"
                     elif hasattr(response, 'candidates') and not response.candidates:
                          error_detail = "応答候補がありません"
                 except Exception as detail_e:
                      print(f"応答詳細の取得エラー: {detail_e}")

                 self.error.emit(f"Gemini APIから無効な応答を受け取りました: {error_detail}")
                 return

            result_text = response.text # 一般的な名前
            self.progress.emit(100) # 完了を示す
            self.finished.emit(result_text)

        except Exception as e:
            # キャンセルと他のエラーを区別する
            if not self._is_interrupted:
                 print(f"Gemini APIエラー (Worker): {e}")
                 self.error.emit(f"Gemini APIからの応答取得に失敗しました: {e}") # 一般的なメッセージ
            # 中断された場合、エラーシグナルはおそらく上記で既に発行されています。
            # キャンセルだった場合に一般的なエラーを発行しないようにします。

    def request_interruption(self):
        """ワーカに操作の中断を要求します。"""
        self._is_interrupted = True
print("Geminiワーカへの中断要求。")


# --- Vimキーバインド付きテキストエディット ---
class VimKeybindTextEdit(QTextEdit):
    """QTextEditをサブクラス化し、基本的なVimナビゲーションキー(h, j, k, l)を追加します。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # キーボードイベントを受け取るために強いフォーカスポリシーを設定
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        cursor = self.textCursor()
        key = event.key()
        scrollbar = self.verticalScrollBar()
        scroll_step = scrollbar.singleStep() # スクロール量を決定 (通常は1行分程度)

        # Shift修飾子やその他の修飾子がないことを確認
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if key == Qt.Key.Key_J:
                # 'j' で下にスクロール
                scrollbar.setValue(scrollbar.value() + scroll_step)
                event.accept()
                return
            elif key == Qt.Key.Key_K:
                # 'k' で上にスクロール
                scrollbar.setValue(scrollbar.value() - scroll_step)
                event.accept()
                return
            elif key == Qt.Key.Key_H:
                # 'h' で左にカーソル移動
                cursor.movePosition(QTextCursor.MoveOperation.Left)
                self.setTextCursor(cursor)
                event.accept()
                return
            elif key == Qt.Key.Key_L:
                # 'l' で右にカーソル移動
                cursor.movePosition(QTextCursor.MoveOperation.Right)
                self.setTextCursor(cursor)
                event.accept()
                return

        # 他のすべてのキープレスはデフォルトの動作に任せる
        super().keyPressEvent(event)


# --- 汎用結果ダイアログ ---
class ResultDialog(QDialog):
    """API呼び出しの結果を表示するシンプルなダイアログ。"""
    def __init__(self, result_text, window_title="結果", parent=None): # window_title を追加
        super().__init__(parent)
        self.setWindowTitle(window_title) # 動的なタイトルを使用
        self.setGeometry(200, 200, 800, 600) # サイズを大きく調整
        self.setMinimumSize(600, 400) # 最小サイズを大きく設定

        layout = QVBoxLayout(self)
        self.text_edit = VimKeybindTextEdit() # カスタムウィジェットを使用
        self.text_edit.setMarkdown(result_text) # setPlainText から setMarkdown に変更
        self.text_edit.setReadOnly(True) # 読み取り専用に戻す
        # フォントサイズを大きくして読みやすくする
        font = self.text_edit.font()
        font.setPointSize(font.pointSize() + 2) # フォントサイズを2ポイント増やす
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit)

        # 一貫性のために標準のボタンボックスを追加
        button_box = QHBoxLayout()
        button_box.addStretch() # ボタンを右に寄せる
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept) # accept() はダイアログを閉じる
        button_box.addWidget(close_button)
        layout.addLayout(button_box)

        # テキストエディットウィジェットに初期フォーカスを設定
        self.text_edit.setFocus()

# --- カスタムプロンプト追加/編集ダイアログ ---
class PromptDialog(QDialog):
    def __init__(self, parent=None, name="", prompt="", shortcut=""):
        super().__init__(parent)
        self.setWindowTitle("カスタムプロンプトの編集")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.name_input = QLineEdit(name)
        self.name_input.setPlaceholderText("ボタンに表示される名前 (例: 要点抽出)")
        form_layout.addRow("名前:", self.name_input)

        self.prompt_input = QTextEdit(prompt)
        self.prompt_input.setPlaceholderText(f"Geminiへの指示を入力します。\n抽出したテキストは {CUSTOM_PROMPT_PLACEHOLDER} の部分に挿入されます。")
        self.prompt_input.setMinimumHeight(100) # 高さを確保
        form_layout.addRow("プロンプト:", self.prompt_input)

        # ショートカット入力 (オプション) - QKeySequenceEditの方が望ましいが、シンプルにQLineEditを使用
        self.shortcut_input = QLineEdit(shortcut)
        self.shortcut_input.setPlaceholderText("例: Ctrl+Shift+A (空欄可)")
        # TODO: QKeySequenceEdit を使用するか、入力検証を追加する
        form_layout.addRow("ショートカット:", self.shortcut_input)


        layout.addLayout(form_layout)

        # OK/キャンセルボタン
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_data(self):
        """ダイアログから入力データを取得します。"""
        name = self.name_input.text().strip()
        prompt = self.prompt_input.toPlainText().strip()
        shortcut = self.shortcut_input.text().strip() # TODO: QKeySequenceに変換/検証
        return name, prompt, shortcut

    def accept(self):
        """OKボタンが押されたときのバリデーション。"""
        name, prompt, _ = self.get_data()
        if not name:
            QMessageBox.warning(self, "入力エラー", "名前を入力してください。")
            return
        if not prompt:
            QMessageBox.warning(self, "入力エラー", "プロンプトを入力してください。")
            return
        if CUSTOM_PROMPT_PLACEHOLDER not in prompt:
             reply = QMessageBox.question(self, "確認",
                                          f"プロンプトに `{CUSTOM_PROMPT_PLACEHOLDER}` が含まれていません。\n"
                                          f"PDFのテキストはこのプレースホルダーの部分に挿入されます。\n"
                                          f"このままでよろしいですか？",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.No:
                 return # ダイアログを開いたままにする
        # TODO: ショートカットの検証 (QKeySequence.fromString)

        super().accept() # バリデーションOKならダイアログを閉じる


# --- PDF画像用のカスタムクリック可能ラベル ---
class ClickableImageLabel(QLabel):
    """左/右クリックのシグナルを発行するQLabelサブクラス。"""
    leftClicked = pyqtSignal()
    rightClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """マウスクリックイベントを処理して左/右クリックを検出します。"""
        if event.button() == Qt.MouseButton.LeftButton:
            width = self.width()
            if width > 0: # ラベルがまだ幅を持たない場合にゼロ除算を回避
                click_x = event.position().x()
                if click_x < width / 2:
                    self.leftClicked.emit()
                else:
                    self.rightClicked.emit()
        # 必要であればイベントを基底クラスに渡すが、ここではおそらく不要
        # super().mousePressEvent(event)


# --- メインアプリケーションウィンドウ ---
class PDFViewer(QMainWindow):
    def __init__(self, initial_file_path=None): # 初期ファイルパス引数を追加
        super().__init__()
        self.settings = QSettings("MyCompany", "PdfViewerGemini") # 組織名, アプリ名
        self.result_dialog = None # 結果ダイアログへの参照（汎用化）
        self.genai_model = None # Geminiモデルインスタンスを格納
        self.doc = None         # 現在のfitz.Document
        self.current_page = 0   # 現在のページインデックス（見開きモードでは左ページ）
        self.zoom_factor = 1.0  # 手動ズームレベル
        self.fit_mode = None    # None, 'width', または 'height'
        self.two_page_mode = False # 見開き表示フラグ
        self.last_viewport_size = None # リサイズ最適化のためのビューポートサイズ追跡
        self.api_call_thread = None # API呼び出しQThreadへの参照（汎用化）
        self.api_call_worker = None # GeminiWorkerへの参照（汎用化）
        self.progress_dialog = None # QProgressDialogへの参照
        self.current_action_name = None # 進行中のアクション名（例："翻訳"）を格納
        self.custom_prompts = [] # カスタムプロンプトを格納するリスト
        self.custom_actions = {} # ツールバー上のカスタムアクションへの参照 (キーはプロンプト名)
        self.spacer_action = None # スペーサーアクションへの参照

        self.setWindowTitle("pdf_viewer_gemini") # タイトルを設定
        self.setGeometry(100, 100, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        # 設定をロードし、最初にGeminiを設定
        self._load_settings()
        self._configure_gemini() # ロードされた設定に基づいて設定

        self._init_ui() # UI要素を作成
        self._populate_settings_ui() # 設定UIにロードされた値を入力
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # ウィンドウがキープレスを受け取るようにする

        # 必要に応じて特定のスタイルを適用（オプション）
        # QApplication.setStyle("Fusion")

        # 初期ファイルパスが指定されている場合、UI表示後に開くようにタイマーを設定
        if initial_file_path:
            # QTimer.singleShotを使用して、イベントループ開始後にファイルを開く
            QTimer.singleShot(0, lambda: self._open_initial_file(initial_file_path))

    def _load_settings(self):
        """QSettingsから設定をロードします。"""
        self.api_key = self.settings.value(SETTINGS_API_KEY, "")
        self.selected_model = self.settings.value(SETTINGS_MODEL, DEFAULT_MODEL)
        # ロードされたモデルが実際に利用可能であることを確認し、利用不可ならフォールバック
        if self.selected_model not in AVAILABLE_MODELS:
            print(f"警告: 保存されたモデル '{self.selected_model}' は利用可能なモデルリストに見つかりません。デフォルト '{DEFAULT_MODEL}' にフォールバックします。")
            self.selected_model = DEFAULT_MODEL
            # オプションで設定ファイルをフォールバックで更新
            self.settings.setValue(SETTINGS_MODEL, self.selected_model)

        # カスタムプロンプトをロード
        try:
            # QSettingsは文字列として保存するため、JSONとして解析する必要がある
            prompts_json = self.settings.value(SETTINGS_CUSTOM_PROMPTS, "[]") # デフォルトは空のJSON配列文字列
            loaded_prompts = json.loads(prompts_json)
            # 簡単な検証 (リストであり、各要素が辞書であることを確認)
            if isinstance(loaded_prompts, list) and all(isinstance(p, dict) for p in loaded_prompts):
                 # 必須キーの存在を確認 (オプションだが推奨)
                 self.custom_prompts = [
                     p for p in loaded_prompts
                     if 'name' in p and 'prompt' in p # 'shortcut' はオプション
                 ]
                 if len(self.custom_prompts) != len(loaded_prompts):
                      print("警告: 一部のカスタムプロンプトが無効な形式のためロードされませんでした。")
            else:
                 print("警告: 保存されたカスタムプロンプトの形式が無効です。空のリストを使用します。")
                 self.custom_prompts = []
                 self.settings.setValue(SETTINGS_CUSTOM_PROMPTS, "[]") # 無効な設定をリセット

        except json.JSONDecodeError:
            print("警告: カスタムプロンプト設定の解析に失敗しました。空のリストを使用します。")
            self.custom_prompts = []
            self.settings.setValue(SETTINGS_CUSTOM_PROMPTS, "[]") # 無効な設定をリセット
        except Exception as e:
             print(f"カスタムプロンプトのロード中に予期せぬエラーが発生しました: {e}")
             self.custom_prompts = []
             # ここでは設定をリセットしない方が安全かもしれない

    def _save_custom_prompts(self):
        """現在のカスタムプロンプトを設定に保存します。"""
        try:
            prompts_json = json.dumps(self.custom_prompts, ensure_ascii=False, indent=2) # インデントして保存
            self.settings.setValue(SETTINGS_CUSTOM_PROMPTS, prompts_json)
            self.settings.sync() # 即時書き込み
            print("カスタムプロンプトを保存しました。")
        except Exception as e:
            print(f"カスタムプロンプトの保存エラー: {e}")
            QMessageBox.critical(self, "保存エラー", f"カスタムプロンプトの保存中にエラーが発生しました:\n{e}")


    def _configure_gemini(self):
        """現在の設定に基づいてGeminiモデルを設定します。"""
        self.genai_model = None # まずモデルをリセット
        # APIキーが設定されていない場合
        if not self.api_key:
            print("警告: Gemini APIキーが設定されていません。翻訳は機能しません。")
            return

        try:
            genai.configure(api_key=self.api_key)
            # 安全性オプションを追加（オプション、必要に応じて調整）
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            self.genai_model = genai.GenerativeModel(
                self.selected_model,
                safety_settings=safety_settings # 安全性設定を適用
            )
            print(f"Geminiが正常に設定されました。モデル: {self.selected_model}")
            if hasattr(self, 'statusBar'): # ステータスバーがまだ存在するか確認
                self.statusBar().showMessage(f"Gemini準備完了 (モデル: {self.selected_model})", STATUS_BAR_MSG_DURATION_MS)
        except Exception as e:
            print(f"Gemini設定エラー: {e}")
            self.genai_model = None
            # UIが完全に初期化されている場合のみQMessageBoxを使用
            if hasattr(self, 'tab_widget'):
                 QMessageBox.warning(self, "Geminiエラー", f"提供されたAPIキー/モデルでGemini APIの設定に失敗しました: {e}\n翻訳機能は無効になります。設定を確認してください。")
            else:
                 # UIが準備できていない場合は、エラーを出力するだけ
                 print(f"重大なGeminiセットアップエラー (UI準備未完了): {e}")


    def _init_ui(self):
        """UI要素を初期化します。"""
        # --- ツールバー ---
        self.main_toolbar = self.addToolBar("ファイル") # ツールバーへの参照を保存
        self.main_toolbar.setObjectName("MainToolBar") # オブジェクト名を設定
        self.main_toolbar.setIconSize(ICON_SIZE) # アイコンサイズ設定を再追加
        self.main_toolbar.setMovable(False) # ツールバーの分離を禁止

        # 開くアクション
        open_action = QAction(get_icon("document-open", "icons/open.png"), "開く (&O)", self) # アイコンを復元
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_pdf)
        self.main_toolbar.addAction(open_action)

        self.main_toolbar.addSeparator()

        # ズームアクション
        zoom_in_action = QAction(get_icon("zoom-in", "icons/zoom-in.png"), "ズームイン (+)", self) # アイコンを復元
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        self.main_toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction(get_icon("zoom-out", "icons/zoom-out.png"), "ズームアウト (-)", self) # アイコンを復元
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        self.main_toolbar.addAction(zoom_out_action)

        # フィットアクション
        fit_width_action = QAction(get_icon("view-fit-width", "icons/fit-width.png"), "幅に合わせる", self) # アイコンを復元
        fit_width_action.setShortcut("Ctrl+W")
        fit_width_action.triggered.connect(self.set_fit_width)
        self.main_toolbar.addAction(fit_width_action)

        fit_height_action = QAction(get_icon("view-fit-height", "icons/fit-height.png"), "高さに合わせる", self) # アイコンを復元
        fit_height_action.setShortcut("Ctrl+H")
        fit_height_action.triggered.connect(self.set_fit_height)
        self.main_toolbar.addAction(fit_height_action)

        self.main_toolbar.addSeparator()

        # 見開き表示切り替え
        self.two_page_action = QAction(get_icon("view-dual", "icons/two-page.png"), "見開き表示", self) # アイコンを復元
        self.two_page_action.setCheckable(True)
        self.two_page_action.setShortcut("Ctrl+T")
        self.two_page_action.triggered.connect(self.toggle_two_page_mode)
        self.main_toolbar.addAction(self.two_page_action)

        self.main_toolbar.addSeparator()

        # ナビゲーションアクション
        prev_action = QAction(get_icon("go-previous", "icons/prev.png"), "前のページ (←)", self) # アイコンを復元
        prev_action.setShortcut(Qt.Key.Key_Left)
        prev_action.triggered.connect(self.prev_page)
        self.main_toolbar.addAction(prev_action)

        next_action = QAction(get_icon("go-next", "icons/next.png"), "次のページ (→)", self) # アイコンを復元
        next_action.setShortcut(Qt.Key.Key_Right)
        next_action.triggered.connect(self.next_page)
        self.main_toolbar.addAction(next_action)

        self.main_toolbar.addSeparator()

        # 翻訳アクション (テキストのみ)
        translate_action = QAction("翻訳 (Tab)", self) # このアクションにはアイコンを削除
        translate_action.setShortcut(Qt.Key.Key_Tab) # Tabショートカットを保持
        translate_action.triggered.connect(self.translate_current_page)
        self.main_toolbar.addAction(translate_action)

        # 意訳アクション (テキストのみ) - 翻訳の隣に移動
        free_translate_action = QAction("意訳 (Ctrl+I)", self)
        free_translate_action.setShortcut("Ctrl+I")
        free_translate_action.triggered.connect(self.free_translate_current_page)
        self.main_toolbar.addAction(free_translate_action)

        # 要約アクション (テキストのみ)
        summarize_action = QAction("要約 (Ctrl+S)", self) # このアクションにはアイコンを削除
        summarize_action.setShortcut("Ctrl+S")
        summarize_action.triggered.connect(self.summarize_current_page)
        self.main_toolbar.addAction(summarize_action)

        # 具体例アクション (テキストのみ)
        example_action = QAction("具体例 (Ctrl+E)", self) # このアクションにはアイコンを削除
        example_action.setShortcut("Ctrl+E")
        example_action.triggered.connect(self.get_example_for_page)
        self.main_toolbar.addAction(example_action)

        # 用語説明アクション (テキストのみ)
        explain_action = QAction("用語説明 (Ctrl+X)", self) # このアクションにはアイコンを削除
        explain_action.setShortcut("Ctrl+X")
        explain_action.triggered.connect(self.explain_term_on_page)
        self.main_toolbar.addAction(explain_action)

        # 再構築アクション (テキストのみ)
        reconstruct_action = QAction("再構築 (Ctrl+R)", self) # 新しいアクションを追加
        reconstruct_action.setShortcut("Ctrl+R") # ショートカットを設定 (例: Ctrl+R)
        reconstruct_action.triggered.connect(self.reconstruct_current_page) # 新しいメソッドに接続
        self.main_toolbar.addAction(reconstruct_action)

        # 解釈アクション (テキストのみ)
        interpret_action = QAction("解釈 (Ctrl+P)", self) # 新しいアクションを追加
        interpret_action.setShortcut("Ctrl+P") # ショートカットを設定 (例: Ctrl+P)
        interpret_action.triggered.connect(self.interpret_current_page) # 新しいメソッドに接続
        self.main_toolbar.addAction(interpret_action)


        # ページラベルを右に寄せるためのスペーサー
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.spacer_action = self.main_toolbar.addWidget(spacer) # スペーサーのアクションを保存

        # ページ番号表示 (ツールバー内)
        self.page_label_toolbar = QLabel("ページ: - / -")
        self.page_label_toolbar.setStyleSheet("padding: 5px; border: none;") # 潜在的な境界線を削除
        # 最大可能ページ番号テキストに基づいて必要な幅を推定 (コメントアウト: 現在は不要)
        # font_metrics = QFontMetrics(self.page_label_toolbar.font())
        # estimated_width = font_metrics.horizontalAdvance("ページ: 9999 / 9999") + 20 # パディングを追加
        # self.page_label_toolbar.setMinimumWidth(estimated_width)
        self.main_toolbar.addWidget(self.page_label_toolbar) # self.main_toolbar を使用


        # --- タブウィジェット ---
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # --- PDFビューワータブ ---
        pdf_view_widget = QWidget()
        pdf_view_layout = QVBoxLayout(pdf_view_widget)
        pdf_view_layout.setContentsMargins(0, 0, 0, 0)
        pdf_view_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.ColorRole.Dark) # 背景色を暗く設定
        self.scroll_area.setWidgetResizable(True) # ウィジェットのリサイズを有効化
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter) # コンテンツを中央揃え
        pdf_view_layout.addWidget(self.scroll_area)

        # カスタムのClickableImageLabelを使用
        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # 画像を中央揃え
        # スクロールエリア内でのより良いスケーリングのために Ignored サイズポリシーを使用
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setBackgroundRole(QPalette.ColorRole.Dark) # スクロールエリアの背景色に合わせる
        self.scroll_area.setWidget(self.image_label)

        # クリックシグナルをナビゲーションメソッドに接続
        self.image_label.leftClicked.connect(self.prev_page)
        self.image_label.rightClicked.connect(self.next_page)


        self.tab_widget.addTab(pdf_view_widget, "ビューワー")

        # --- 設定タブ ---
        settings_widget = QWidget()
        settings_outer_layout = QVBoxLayout(settings_widget) # 全体構造にQVBoxLayoutを使用
        settings_form_layout = QFormLayout() # ラベルとフィールドのペアにFormLayoutを使用

        # APIキー
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("ここにGemini APIキーを入力") # プレースホルダーテキスト
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password) # パスワード入力モード
        save_api_key_button = QPushButton("APIキーを保存")
        save_api_key_button.clicked.connect(self._save_api_key)
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(self.api_key_input)
        api_key_layout.addWidget(save_api_key_button)
        settings_form_layout.addRow("Gemini APIキー:", api_key_layout)

        # モデル選択
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.setToolTip("使用するGeminiのモデルを選択してください。") # ツールチップ
        save_model_button = QPushButton("モデルを保存")
        save_model_button.clicked.connect(self._save_selected_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(save_model_button)
        settings_form_layout.addRow("利用モデル:", model_layout) # ラベル名を変更

        # フォームレイアウトを外側のレイアウトに追加
        settings_outer_layout.addLayout(settings_form_layout)

        # --- カスタムプロンプト管理セクション ---
        settings_outer_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)) # スペーサーを追加
        custom_prompts_label = QLabel("カスタムプロンプト:")
        settings_outer_layout.addWidget(custom_prompts_label)

        self.custom_prompt_list = QListWidget()
        self.custom_prompt_list.setToolTip("ダブルクリックで編集、選択して削除")
        self.custom_prompt_list.itemDoubleClicked.connect(self._edit_custom_prompt) # ダブルクリックで編集
        settings_outer_layout.addWidget(self.custom_prompt_list)

        custom_button_layout = QHBoxLayout()
        self.add_prompt_button = QPushButton("追加")
        self.add_prompt_button.setIcon(get_icon("list-add"))
        self.add_prompt_button.clicked.connect(self._add_custom_prompt)
        custom_button_layout.addWidget(self.add_prompt_button)

        self.edit_prompt_button = QPushButton("編集")
        self.edit_prompt_button.setIcon(get_icon("document-edit"))
        self.edit_prompt_button.clicked.connect(self._edit_custom_prompt)
        self.edit_prompt_button.setEnabled(False) # 最初は無効
        custom_button_layout.addWidget(self.edit_prompt_button)

        self.remove_prompt_button = QPushButton("削除")
        self.remove_prompt_button.setIcon(get_icon("list-remove"))
        self.remove_prompt_button.clicked.connect(self._remove_custom_prompt)
        self.remove_prompt_button.setEnabled(False) # 最初は無効
        custom_button_layout.addWidget(self.remove_prompt_button)
        custom_button_layout.addStretch() # ボタンを左に寄せる
        settings_outer_layout.addLayout(custom_button_layout)

        # リストの選択変更時に編集/削除ボタンの有効/無効を切り替え
        self.custom_prompt_list.itemSelectionChanged.connect(self._update_prompt_edit_buttons_state)


        # 設定を上部に押し上げるためのスペーサー
        settings_outer_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.tab_widget.addTab(settings_widget, "設定")

        # --- ステータスバー ---
        self.statusBar().showMessage("準備完了")

    def _populate_settings_ui(self):
        """QSettingsからロードされた値で設定UIを埋めます。"""
        self.api_key_input.setText(self.api_key)

        # コンボボックスを保存されたモデルに設定
        index = self.model_combo.findText(self.selected_model, Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        else:
            # 保存されたモデルが見つからない場合、デフォルトを選択
            index = self.model_combo.findText(DEFAULT_MODEL, Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                # デフォルトすら見つからない場合（現在のロジックでは起こらないはず）、最初のアイテムを選択
                self.model_combo.setCurrentIndex(0)

        # カスタムプロンプトUIを更新
        self._update_custom_prompt_list_widget()
        self._update_toolbar_custom_buttons() # ツールバー更新メソッドを呼び出す


    def _save_api_key(self):
        """入力フィールドのAPIキーを設定に保存します。"""
        new_key = self.api_key_input.text().strip()
        if new_key:
            if new_key != self.api_key: # 変更された場合のみ保存して再設定
                self.api_key = new_key
                self.settings.setValue(SETTINGS_API_KEY, self.api_key)
                self.settings.sync() # 即時書き込みを保証 (オプション)
                self._configure_gemini() # 新しいキーで再設定
                self.statusBar().showMessage("APIキーを保存しました。", STATUS_BAR_MSG_DURATION_MS)
            else:
                 # 変更されていない場合
                 self.statusBar().showMessage("APIキーは変更されていません。", STATUS_BAR_MSG_DURATION_MS)
        else:
            # 空の場合
            QMessageBox.warning(self, "APIキー保存エラー", "APIキーを空にすることはできません。")

    def _save_selected_model(self):
        """コンボボックスから選択されたモデルを設定に保存します。"""
        new_model = self.model_combo.currentText()
        if new_model:
            if new_model != self.selected_model: # 変更された場合のみ保存して再設定
                self.selected_model = new_model
                self.settings.setValue(SETTINGS_MODEL, self.selected_model)
                self.settings.sync() # 即時書き込みを保証 (オプション)
                self._configure_gemini() # 新しいモデルで再設定
                self.statusBar().showMessage(f"モデル '{new_model}' を保存しました。", STATUS_BAR_MSG_DURATION_MS)
            else:
                # 変更されていない場合
                self.statusBar().showMessage(f"モデル '{new_model}' は既に選択されています。", STATUS_BAR_MSG_DURATION_MS)

    # --- カスタムプロンプト管理メソッド ---

    def _update_custom_prompt_list_widget(self):
        """設定リストウィジェットを現在のカスタムプロンプトで更新します。"""
        self.custom_prompt_list.clear()
        for prompt_data in self.custom_prompts:
            name = prompt_data.get('name', '無名プロンプト')
            shortcut = prompt_data.get('shortcut')
            display_text = f"{name}"
            if shortcut:
                display_text += f" ({shortcut})"
            item = QListWidgetItem(display_text)
            # アイテムにプロンプトデータ全体を関連付ける (編集/削除時に使用)
            item.setData(Qt.ItemDataRole.UserRole, prompt_data)
            self.custom_prompt_list.addItem(item)
        self._update_prompt_edit_buttons_state() # ボタンの状態も更新

    def _update_toolbar_custom_buttons(self):
        """ツールバー上のカスタムプロンプトアクションを更新します。"""
        # 保存したツールバー参照を使用
        if not hasattr(self, 'main_toolbar') or not self.main_toolbar:
            print("エラー: メインツールバーが見つかりません。カスタムアクションを追加できません。")
            return
        # スペーサーアクションが初期化されているか確認
        if not hasattr(self, 'spacer_action') or not self.spacer_action:
             print("エラー: スペーサーアクションが見つかりません。カスタムアクションを正しい位置に追加できません。")
             # フォールバックとしてツールバーの最後に追加することも検討できるが、まずはエラーを報告
             return

        # 既存のカスタムアクションを削除
        for name, action in self.custom_actions.items():
            self.main_toolbar.removeAction(action)
        self.custom_actions.clear()

        # 新しいアクションを追加 (スペーサーの前に追加)
        for prompt_data in reversed(self.custom_prompts): # 逆順に追加して正しい順序にする
            name = prompt_data.get('name')
            prompt_template = prompt_data.get('prompt')
            shortcut_str = prompt_data.get('shortcut', '')

            if not name or not prompt_template:
                continue # 無効なデータはスキップ

            action = QAction(name, self)
            # functools.partial を使用して、トリガー時に関数に引数を渡す
            from functools import partial
            action.triggered.connect(partial(self._execute_custom_prompt, name, prompt_template))

            # ツールチップにプロンプト内容を表示
            action.setToolTip(f"プロンプト:\n{prompt_template}")

            # ショートカットを設定 (エラーハンドリング付き)
            if shortcut_str:
                try:
                    key_sequence = QKeySequence.fromString(shortcut_str, QKeySequence.SequenceFormat.PortableText)
                    if not key_sequence.isEmpty():
                        action.setShortcut(key_sequence)
                    else:
                         print(f"警告: プロンプト '{name}' のショートカット '{shortcut_str}' は無効です。")
                except Exception as e:
                     print(f"警告: プロンプト '{name}' のショートカット '{shortcut_str}' の設定中にエラー: {e}")

            # スペーサーのアクションの前にアクションを挿入
            self.main_toolbar.insertAction(self.spacer_action, action)
            self.custom_actions[name] = action # アクション参照を保存


    def _update_prompt_edit_buttons_state(self):
        """リストの選択に基づいて編集/削除ボタンの有効/無効を切り替えます。"""
        selected_items = self.custom_prompt_list.selectedItems()
        is_selected = bool(selected_items)
        self.edit_prompt_button.setEnabled(is_selected)
        self.remove_prompt_button.setEnabled(is_selected)

    def _add_custom_prompt(self):
        """新しいカスタムプロンプトを追加するためのダイアログを表示します。"""
        dialog = PromptDialog(self)
        if dialog.exec(): # exec() はモーダル表示し、ユーザーがOK/キャンセルを押すまで待機
            name, prompt, shortcut = dialog.get_data()
            # 名前が既存のものと重複していないかチェック (オプション)
            if any(p['name'] == name for p in self.custom_prompts):
                 QMessageBox.warning(self, "追加エラー", f"名前 '{name}' のプロンプトは既に存在します。")
                 return

            new_prompt_data = {'name': name, 'prompt': prompt}
            if shortcut: # ショートカットが空でなければ追加
                 new_prompt_data['shortcut'] = shortcut

            self.custom_prompts.append(new_prompt_data)
            self._save_custom_prompts()
            self._update_custom_prompt_list_widget()
            self._update_toolbar_custom_buttons() # ツールバーを更新
            self.statusBar().showMessage(f"カスタムプロンプト '{name}' を追加しました。", STATUS_BAR_MSG_DURATION_MS)

    def _edit_custom_prompt(self):
        """選択されているカスタムプロンプトを編集するためのダイアログを表示します。"""
        selected_items = self.custom_prompt_list.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        original_data = item.data(Qt.ItemDataRole.UserRole)
        if not original_data:
             QMessageBox.warning(self, "編集エラー", "選択されたアイテムからプロンプトデータを取得できませんでした。")
             return

        original_name = original_data.get('name', '')
        original_prompt = original_data.get('prompt', '')
        original_shortcut = original_data.get('shortcut', '')

        dialog = PromptDialog(self, name=original_name, prompt=original_prompt, shortcut=original_shortcut)
        if dialog.exec():
            new_name, new_prompt, new_shortcut = dialog.get_data()

            # 名前が変更され、かつ新しい名前が *他の* 既存プロンプトと重複していないかチェック
            if new_name != original_name and any(p['name'] == new_name for p in self.custom_prompts):
                 QMessageBox.warning(self, "編集エラー", f"名前 '{new_name}' のプロンプトは既に存在します。")
                 return

            # 元のデータをリストから見つけて更新
            for i, p_data in enumerate(self.custom_prompts):
                 if p_data.get('name') == original_name: # 名前で識別
                     self.custom_prompts[i]['name'] = new_name
                     self.custom_prompts[i]['prompt'] = new_prompt
                     if new_shortcut:
                         self.custom_prompts[i]['shortcut'] = new_shortcut
                     elif 'shortcut' in self.custom_prompts[i]:
                          del self.custom_prompts[i]['shortcut'] # ショートカットが削除された場合
                     break # 更新したらループを抜ける

            self._save_custom_prompts()
            self._update_custom_prompt_list_widget()
            self._update_toolbar_custom_buttons() # ツールバーを更新
            self.statusBar().showMessage(f"カスタムプロンプト '{new_name}' を更新しました。", STATUS_BAR_MSG_DURATION_MS)


    def _remove_custom_prompt(self):
        """選択されているカスタムプロンプトを削除します。"""
        selected_items = self.custom_prompt_list.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        prompt_data = item.data(Qt.ItemDataRole.UserRole)
        if not prompt_data:
             QMessageBox.warning(self, "削除エラー", "選択されたアイテムからプロンプトデータを取得できませんでした。")
             return

        name_to_remove = prompt_data.get('name', '不明なプロンプト')

        reply = QMessageBox.question(self, "削除の確認",
                                     f"カスタムプロンプト '{name_to_remove}' を削除しますか？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # 名前でリストから削除
            self.custom_prompts = [p for p in self.custom_prompts if p.get('name') != name_to_remove]
            self._save_custom_prompts()
            self._update_custom_prompt_list_widget()
            self._update_toolbar_custom_buttons() # ツールバーを更新
            self.statusBar().showMessage(f"カスタムプロンプト '{name_to_remove}' を削除しました。", STATUS_BAR_MSG_DURATION_MS)

    def _execute_custom_prompt(self, name, prompt_template):
        """指定されたカスタムプロンプトを実行します。"""
        print(f"カスタムプロンプト '{name}' を実行します。")
        # result_dialog_title はプロンプト名を使用
        self._call_gemini_api(prompt_template, f"{name} 結果", name)


    # --- ファイル操作メソッド ---
    def open_pdf(self):
        """ユーザーが選択したPDFファイルを開きます。"""
        # 最後に開いたディレクトリを取得
        last_dir = self.settings.value(SETTINGS_LAST_DIR, os.path.expanduser("~")) # デフォルトはホーム

        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDFファイルを開く", last_dir, "PDFファイル (*.pdf);;すべてのファイル (*)"
        )
        if file_path:
             # 開いたファイルのディレクトリを保存
             current_dir = os.path.dirname(file_path)
             self.settings.setValue(SETTINGS_LAST_DIR, current_dir)

             # 以前のドキュメントを安全に閉じる
             self._close_current_doc()

             try:
                self.doc = fitz.open(file_path)
                if len(self.doc) == 0:
                    raise ValueError("PDFファイルにページが含まれていません。") # 空のPDFを処理

                self.current_page = 0
                self.zoom_factor = 1.0 # 手動ズームをリセット
                self.fit_mode = 'height' # デフォルトは高さに合わせる
                self.two_page_mode = False # デフォルトは単一ページ
                self.two_page_action.setChecked(False) # トグルボタンの状態を更新
                self.last_viewport_size = None # ビューポート追跡をリセット

                # QTimerを使用して、初期表示のためにビューポートサイズが利用可能であることを確認
                QTimer.singleShot(0, self.display_page)
                self.setWindowTitle(f"{os.path.basename(file_path)} - PDFviewer") # ファイル名を最初に表示

             except Exception as e:
                print(f"PDF '{file_path}' のオープンエラー: {e}")
                QMessageBox.critical(self, "PDFオープンエラー", f"PDFファイルを開けませんでした:\n{e}")
                self._reset_viewer_state() # エラー時にUI要素をリセット

    def _open_initial_file(self, file_path):
        """起動時に指定されたファイルを開きます。"""
        print(f"コマンドライン引数からファイルを開こうとしています: {file_path}")
        if file_path and os.path.exists(file_path) and file_path.lower().endswith(".pdf"):
            # 以前のドキュメントを安全に閉じる
            self._close_current_doc()
            try:
                self.doc = fitz.open(file_path)
                if len(self.doc) == 0:
                    raise ValueError("PDFファイルにページが含まれていません。")

                self.current_page = 0
                self.zoom_factor = 1.0
                self.fit_mode = 'height'
                self.two_page_mode = False
                self.two_page_action.setChecked(False)
                self.last_viewport_size = None

                # display_pageを直接呼び出す (タイマー内なのでUIは準備完了のはず)
                self.display_page()
                self.setWindowTitle(f"{os.path.basename(file_path)} - PdfViewerGemini") # ファイル名を最初に表示
                # 最後に開いたディレクトリも更新
                current_dir = os.path.dirname(file_path)
                self.settings.setValue(SETTINGS_LAST_DIR, current_dir)
                print(f"コマンドライン引数のファイルを開きました: {file_path}")

            except Exception as e:
                print(f"初期ファイル '{file_path}' のオープンエラー: {e}")
                QMessageBox.critical(self, "初期ファイルオープンエラー", f"指定されたPDFファイルを開けませんでした:\n{file_path}\n\nエラー: {e}")
                self._reset_viewer_state()
        else:
            # ファイルが存在しないか、PDFでない場合
            msg = f"指定されたファイルが見つからないか、PDFファイルではありません:\n{file_path}"
            print(msg)
            QMessageBox.warning(self, "ファイルオープンエラー", msg)
            # ここではリセットは不要かもしれないが、念のため
            self._reset_viewer_state()


    def _close_current_doc(self):
        """現在開いているドキュメントを安全に閉じます。"""
        if self.doc:
            try:
                self.doc.close()
                print("以前のPDFドキュメントを閉じました。")
            except Exception as e:
                print(f"以前のPDFドキュメントのクローズエラー: {e}")
            finally:
                self.doc = None

    def _reset_viewer_state(self):
         """ビューワーのUI要素をデフォルト状態にリセットします。"""
         self._close_current_doc() # ドキュメントが閉じられていることを確認
         self.image_label.clear()
         self.image_label.setText("PDFを開いてください") # プレースホルダーテキスト
         self.image_label.resize(self.image_label.sizeHint()) # ラベルサイズをコンテンツに合わせる
         self.page_label_toolbar.setText("ページ: - / -")
         self.setWindowTitle("PdfViewerGemini") # タイトルをリセット
         self.fit_mode = None
         self.two_page_mode = False
         self.two_page_action.setChecked(False)
         self.last_viewport_size = None
         self.current_page = 0
         self.zoom_factor = 1.0


    def display_page(self):
        """現在のページ（複数可）をレンダリングして表示します。"""
        # ドキュメントが無効か、ページが範囲外の場合にリセット
        if not self.doc or not (0 <= self.current_page < len(self.doc)):
            self._reset_viewer_state()
            # ドキュメントが存在し、ページがある場合、総ページ数を表示
            if self.doc and len(self.doc) > 0:
                 self.page_label_toolbar.setText(f"ページ: ? / {len(self.doc)}")
            return

        try:
            current_zoom = self.zoom_factor # 基本の手動ズーム
            page_num_text = f"{self.current_page + 1}"

            # --- ロードするページを決定し、結合された寸法を計算 ---
            indices_to_load = [self.current_page]
            is_two_page_view = False
            # モードが有効で、0ページ目ではなく、次のページが存在する場合にのみ2ページを表示
            if self.two_page_mode and self.current_page > 0 and self.current_page + 1 < len(self.doc):
                 indices_to_load.append(self.current_page + 1)
                 page_num_text = f"{self.current_page + 1}-{self.current_page + 2}"
                 is_two_page_view = True

            # ズーム計算のために寸法を取得するためにページをロード
            loaded_pages = [self.doc.load_page(p_idx) for p_idx in indices_to_load]
            # 最初のチェックが通れば起こらないはずだが、安全のために
            if not loaded_pages:
                 print("エラー: 表示用にロードされたページがありません。")
                 return

            total_unzoomed_width = sum(p.rect.width for p in loaded_pages)
            max_unzoomed_height = max(p.rect.height for p in loaded_pages)

            # ズーム計算 *前* に、見開き表示での幅計算用の基本スペーシングを追加
            spacing = TWO_PAGE_SPACING_BASE if is_two_page_view else 0
            total_unzoomed_width_with_spacing = total_unzoomed_width + spacing

            # --- フィットモードに基づいてズーム係数を計算 ---
            viewport = self.scroll_area.viewport() # ビューポートウィジェットを取得
            viewport_width = viewport.width()
            viewport_height = viewport.height()

            if self.fit_mode == 'width':
                available_width = max(1, viewport_width - FIT_PADDING) # パディングを考慮
                if total_unzoomed_width_with_spacing > 0:
                     current_zoom = available_width / total_unzoomed_width_with_spacing
                else: current_zoom = 1.0 # ゼロ除算を回避
            elif self.fit_mode == 'height':
                 available_height = max(1, viewport_height - FIT_PADDING) # パディングを考慮
                 if max_unzoomed_height > 0:
                     current_zoom = available_height / max_unzoomed_height
                 else: current_zoom = 1.0
            # fit_modeがNoneの場合、current_zoomはself.zoom_factor（手動ズーム）のまま

            # ズーム係数を制限内にクランプ
            current_zoom = max(MIN_ZOOM, min(MAX_ZOOM, current_zoom))

            zoom_matrix = fitz.Matrix(current_zoom, current_zoom)
            scaled_spacing = int(spacing * current_zoom) # スペーシングをズームでスケーリング

            # --- ページをレンダリング ---
            pixmaps = []
            for page in loaded_pages:
                 # 必要に応じて高DPIを使用することを検討するが、パフォーマンスに影響する
                 # pix = page.get_pixmap(matrix=zoom_matrix, dpi=150, alpha=False)
                 pix = page.get_pixmap(matrix=zoom_matrix, alpha=False) # 透明度なしでレンダリング
                 # pixmap生成のエラーチェック
                 if pix.width == 0 or pix.height == 0:
                      print(f"警告: ページ {page.number} 用に空のpixmapが生成されました")
                      # 後でエラーを回避するために小さなプレースホルダーpixmapを作成
                      placeholder_img = QImage(10, 10, QImage.Format.Format_RGB888)
                      placeholder_img.fill(Qt.GlobalColor.red) # エラーを視覚的に示す
                      pixmaps.append(QPixmap.fromImage(placeholder_img))
                      continue

                 # QImageに変換 (アルファチャンネルの有無でフォーマットを選択)
                 img_format = QImage.Format.Format_RGB888 if pix.alpha == 0 else QImage.Format.Format_RGBA8888
                 qimage = QImage(pix.samples, pix.width, pix.height, pix.stride, img_format)
                 pixmaps.append(QPixmap.fromImage(qimage))


            # --- 見開き表示の場合、Pixmapsを結合 ---
            if is_two_page_view and len(pixmaps) > 1:
                total_pix_width = sum(p.width() for p in pixmaps) + scaled_spacing
                max_pix_height = max(p.height() for p in pixmaps)

                combined_pixmap = QPixmap(total_pix_width, max_pix_height)
                # 視覚的な一貫性のために背景色で塗りつぶす
                combined_pixmap.fill(self.scroll_area.palette().color(QPalette.ColorRole.Dark))

                painter = QPainter(combined_pixmap)
                current_x = 0
                for i, pix in enumerate(pixmaps):
                    # 高さが異なる場合は垂直方向に中央揃え
                    y_offset = (max_pix_height - pix.height()) // 2
                    painter.drawPixmap(current_x, y_offset, pix)
                    current_x += pix.width()
                    if i == 0: # 最初のページの後ろにスペーシングを追加
                         current_x += scaled_spacing
                painter.end()
                final_pixmap = combined_pixmap
            elif pixmaps: # 単一ページ表示または利用可能なページが1つだけの場合
                final_pixmap = pixmaps[0]
            else:
                # 何か問題が発生した場合
                final_pixmap = QPixmap() # 空のpixmap

            # --- 最終的なPixmapを表示 ---
            if not final_pixmap.isNull():
                # SizePolicyがIgnored/Ignoredの場合、pixmapを設定すると暗黙的にラベルのサイズが決まる
                self.image_label.setPixmap(final_pixmap)
                # オプション: 配置の問題が発生した場合に明示的にリサイズするが、Ignoredポリシーでは通常不要
                # self.image_label.resize(final_pixmap.size())
            else:
                 # pixmapがNullの場合
                 self.image_label.clear()
                 self.image_label.setText(f"ページ {page_num_text} のレンダリングエラー")
                 self.image_label.resize(self.image_label.sizeHint()) # テキストに合わせてリサイズ

            # ページ番号表示を更新
            self.page_label_toolbar.setText(f"ページ: {page_num_text} / {len(self.doc)}")

        except Exception as e:
            print(f"{self.current_page} から始まるページの表示エラー: {e}")
            import traceback
            traceback.print_exc() # デバッグ用に詳細なトレースバックを出力
            self.image_label.setText(f"ページ {page_num_text} の表示エラー:\n{e}")
            self.image_label.resize(self.image_label.sizeHint()) # エラーメッセージに合わせてリサイズ
            self.page_label_toolbar.setText(f"ページ: {page_num_text} / {len(self.doc)} (エラー)")


    # --- Gemini API呼び出しメソッド ---

    def translate_current_page(self):
        """現在表示されているページの翻訳を開始します。"""
        self._call_gemini_api(TRANSLATION_PROMPT_TEMPLATE, "翻訳結果", "翻訳")

    def summarize_current_page(self):
        """現在表示されているページの要約を開始します。"""
        self._call_gemini_api(SUMMARIZE_PROMPT_TEMPLATE, "要約結果", "要約")

    def get_example_for_page(self):
        """現在表示されているページの具体例取得を開始します。"""
        self._call_gemini_api(EXAMPLE_PROMPT_TEMPLATE, "具体例", "具体例生成")

    def explain_term_on_page(self):
        """現在表示されているページの用語説明を開始します。"""
        self._call_gemini_api(EXPLAIN_TERM_PROMPT_TEMPLATE, "用語説明", "用語説明")

    def free_translate_current_page(self):
        """現在表示されているページの意訳を開始します。"""
        self._call_gemini_api(FREE_TRANSLATION_PROMPT_TEMPLATE, "意訳結果", "意訳")

    def reconstruct_current_page(self):
        """現在表示されているページのテキストを再構築します。"""
        self._call_gemini_api(RECONSTRUCT_PROMPT_TEMPLATE, "再構築結果", "再構築")

    def interpret_current_page(self):
        """現在表示されているページのテキストの意図を解釈します。"""
        self._call_gemini_api(INTERPRET_PROMPT_TEMPLATE, "解釈結果", "解釈")


    def _call_gemini_api(self, prompt_template, result_dialog_title, action_name):
        """様々なアクションのためにGemini APIを呼び出す汎用メソッド。"""
        # ドキュメントが開かれていない場合
        if not self.doc:
            QMessageBox.information(self, f"{action_name}不可", "PDFファイルが開かれていません。")
            return
        # Geminiモデルが設定されていない場合
        if not self.genai_model:
             QMessageBox.warning(self, f"{action_name}エラー", f"Geminiモデルが設定されていません。\n設定タブでAPIキーとモデルを確認してください。")
             return

        # 別のAPI呼び出しが既に実行中か確認
        if self.api_call_thread and self.api_call_thread.isRunning():
             # self.current_action_name を使用して実行中のアクションを表示
             running_action = self.current_action_name or "別の操作"
             QMessageBox.information(self, f"{action_name}不可", f"現在、{running_action}が進行中です。\n完了するまでお待ちください。")
             return

        # 終了した（まだ削除されていない）スレッド/ワーカーを最初にクリーンアップ
        self._cleanup_api_call_thread()

        try:
            # 現在のビューに基づいて処理するページを決定
            indices_to_process = [self.current_page]
            if self.two_page_mode and self.current_page > 0 and self.current_page + 1 < len(self.doc):
                 indices_to_process.append(self.current_page + 1)

            # 関連するページからテキストを抽出
            extracted_text = ""
            for page_idx in indices_to_process:
                 # インデックスが有効であることを確認
                 if 0 <= page_idx < len(self.doc):
                    page = self.doc.load_page(page_idx)
                    try:
                        # テキストを抽出 (ソートして読みやすい順序にする)
                        page_text = page.get_text("text", sort=True)
                    except Exception as text_extract_err:
                         # テキスト抽出エラー
                         print(f"ページ {page_idx + 1} からのテキスト抽出エラー: {text_extract_err}")
                         page_text = ""

                    # テキストが存在する場合
                    if page_text:
                        extracted_text += f"[ページ {page_idx + 1}]\n"
                        extracted_text += page_text.strip() + "\n\n"
                 else:
                    # 無効なページインデックス
                    print(f"警告: {action_name} に無効なページインデックス {page_idx} が要求されました。")

            # 抽出されたテキストが空の場合
            if not extracted_text.strip():
                QMessageBox.information(self, f"{action_name}不可", "現在の表示ページからテキストを抽出できませんでした。")
                return

            # --- ワーカースレッド経由でGemini APIの準備と呼び出し ---
            prompt = prompt_template.format(text=extracted_text)
            self.current_action_name = action_name # 現在のアクションを保存

            # プログレスダイアログの設定と表示
            # 既存のプログレスダイアログがあれば閉じる
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

            self.progress_dialog = QProgressDialog(f"Geminiに接続中 ({action_name})...", "キャンセル", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal) # モーダルダイアログ
            self.progress_dialog.setWindowTitle(f"{action_name}中")
            self.progress_dialog.setMinimumDuration(PROGRESS_DIALOG_DELAY_MS) # すぐに表示されないように遅延
            self.progress_dialog.setValue(0)
            self.progress_dialog.canceled.connect(self._cancel_api_call) # キャンセルボタンの接続

            self.statusBar().showMessage(f"{action_name}を開始します...")

            # --- スレッド/ワーカーの作成と開始 ---
            self.api_call_thread = QThread(self)
            self.api_call_worker = GeminiWorker(self.genai_model, prompt) # 汎用化されたワーカーを使用
            self.api_call_worker.moveToThread(self.api_call_thread)

            # シグナル/スロットを接続（汎用名を使用）
            self.api_call_thread.started.connect(self.api_call_worker.run)
            # 結果ダイアログのタイトルを finished ハンドラに渡す
            self.api_call_worker.finished.connect(lambda result: self._handle_api_call_finished(result, result_dialog_title))
            self.api_call_worker.error.connect(self._handle_api_call_error)
            self.api_call_worker.progress.connect(self._update_api_call_progress)

            # クリーンアップのための接続 (スレッド終了時に自動でワーカーとスレッドを削除)
            self.api_call_worker.finished.connect(self.api_call_thread.quit)
            self.api_call_worker.error.connect(self.api_call_thread.quit) # エラー時もスレッドを終了
            self.api_call_thread.finished.connect(self.api_call_worker.deleteLater)
            self.api_call_thread.finished.connect(self.api_call_thread.deleteLater)
            self.api_call_thread.finished.connect(self._clear_api_call_references) # 参照をクリア

            self.api_call_thread.start()

        except Exception as e:
            # 準備中のエラー
            print(f"{action_name} の準備エラー: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, f"{action_name}準備エラー", f"{action_name}の準備中に予期せぬエラーが発生しました:\n{e}")
            # プログレスダイアログがあれば閉じる
            if self.progress_dialog:
                self.progress_dialog.close()
            self.statusBar().showMessage(f"{action_name}準備失敗", STATUS_BAR_MSG_DURATION_MS * 2)
            self._cleanup_api_call_thread() # クリーンアップを試みる


    def _update_api_call_progress(self, value):
        """任意のAPI呼び出しのプログレスダイアログを更新します。"""
        # プログレスダイアログが存在する場合
        if self.progress_dialog:
             action = self.current_action_name or "処理"
             # ダイアログが非表示で、進捗が進んでいる場合（0と100を除く）に表示
             if not self.progress_dialog.isVisible() and value > 0 and value < 100:
                  # minimumDuration を超えていれば表示される
                  pass # show() は自動で呼ばれるため不要
             self.progress_dialog.setValue(value)
             # 進捗に応じてラベルテキストを更新
             if value < 50:
                  # リクエスト送信中
                  self.progress_dialog.setLabelText(f"Geminiにリクエスト送信中 ({action})...")
             elif value < 100 :
                  # 処理中
                  self.progress_dialog.setLabelText(f"Geminiが{action}を処理中...")
             else:
                  # 完了
                  self.progress_dialog.setLabelText(f"{action}完了。")


    def _handle_api_call_finished(self, result_text, window_title):
        """任意のGemini API呼び出しの成功完了を処理します。"""
        action = self.current_action_name or "操作"
        # プログレスダイアログがあれば閉じる
        if self.progress_dialog:
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()
        self.statusBar().showMessage(f"{action}が完了しました。", STATUS_BAR_MSG_DURATION_MS)

        # 新しい結果ダイアログを表示する前に、既存のものを閉じる
        if self.result_dialog and self.result_dialog.isVisible():
            self.result_dialog.close()

        # 結果を非モーダルの汎用ダイアログに表示
        self.result_dialog = ResultDialog(result_text, window_title, self) # 汎用ダイアログを使用
        self.result_dialog.show()

        # 参照はスレッド終了時に _clear_api_call_references でクリアされる

    def _handle_api_call_error(self, error_message):
        """Geminiワーカーによって報告されたエラーを処理します。"""
        action = self.current_action_name or "操作"
        # プログレスダイアログがあれば閉じる
        if self.progress_dialog:
            self.progress_dialog.close()

        # エラーがキャンセルによるものか確認（一般的なメッセージを使用）
        if "キャンセルされました" in error_message:
            self.statusBar().showMessage(f"{action}がキャンセルされました。", STATUS_BAR_MSG_DURATION_MS)
            print(f"{action} が明示的にキャンセルされました。")
        else:
            # その他のエラー
            self.statusBar().showMessage(f"{action}エラーが発生しました。", STATUS_BAR_MSG_DURATION_MS * 2)
            QMessageBox.critical(self, f"{action}エラー", error_message)

        # 参照はスレッド終了時に _clear_api_call_references でクリアされる

    def _cancel_api_call(self):
        """任意のAPI呼び出しのプログレスダイアログからのキャンセル要求を処理します。"""
        action = self.current_action_name or "操作"
        print(f"{action} のキャンセルがプログレスダイアログ経由で要求されました。")
        # ワーカーが存在する場合、中断を要求
        if self.api_call_worker:
            self.api_call_worker.request_interruption()
        # プログレスダイアログが存在する場合、状態を変更
        if self.progress_dialog:
            self.progress_dialog.setLabelText("キャンセル中...")
            self.progress_dialog.setEnabled(False) # キャンセルボタンを無効化

    def _cleanup_api_call_thread(self):
        """終了したAPI呼び出しスレッドとワーカーが存在する場合、それらをクリーンアップします。"""
        # スレッドが存在し、終了している場合
        if self.api_call_thread and self.api_call_thread.isFinished():
            print("終了したAPI呼び出しスレッド/ワーカーをクリーンアップしています。")
            # 削除は finished シグナルに接続された deleteLater によって処理されます
            # ここでは参照をクリアするだけで良い場合が多い
            self._clear_api_call_references()


    def _clear_api_call_references(self):
        """API呼び出しスレッドとワーカーへの参照をクリアします。スレッド終了時に呼び出されます。"""
        print("API呼び出しスレッド/ワーカーの参照をクリアしています。")
        self.api_call_thread = None
        self.api_call_worker = None
        self.current_action_name = None # アクション名をリセット

    # --- ナビゲーションメソッド ---
    def prev_page(self):
        """前のページに移動します。"""
        # ドキュメントが存在し、最初のページでない場合
        if self.doc and self.current_page > 0:
            # 見開きモードで2ページ目以降なら2ページ、そうでなければ1ページ戻る
            delta = 2 if self.two_page_mode and self.current_page > 1 else 1
            # 特殊ケース: 見開きモードで1ページ目にいる場合、戻ると0ページ目（単一表示）になる
            if self.two_page_mode and self.current_page == 1:
                 delta = 1
            new_page = max(0, self.current_page - delta) # 0未満にならないように
            # ページが実際に変更された場合のみ更新
            if new_page != self.current_page:
                self.current_page = new_page
                self.display_page()

    def next_page(self):
        """次のページに移動します。"""
        # ドキュメントが存在する場合
        if self.doc:
            num_pages = len(self.doc)
            # 既に最終ページ（または何らかの理由でそれを超えている）の場合
            if self.current_page >= num_pages - 1:
                 return

            delta = 1 # 基本は1ページ進む
            # 見開きモードの場合
            if self.two_page_mode:
                 # 0ページ目にいる場合、次は1（見開きレイアウトでも概念的には単一表示）
                 if self.current_page == 0:
                     delta = 1
                 # それ以外の場合、2つジャンプするが、最後から2番目のページを超えないようにする
                 # （例：N-2ページ目にいる場合、次はN-1であるべきで、ペアの開始として存在しないNではない）
                 elif self.current_page + 2 < num_pages:
                     delta = 2
                 else: # 最後から2番目のページにいる場合、次は最終ページ (移動量は1)
                      delta = 1


            potential_next_page = self.current_page + delta

            # 最終ページインデックスを超えないようにする
            if potential_next_page < num_pages:
                 self.current_page = potential_next_page
                 self.display_page()


    # --- ズームとフィットメソッド ---
    def zoom_in(self):
        """表示を拡大します。"""
        # ドキュメントが存在する場合
        if self.doc:
            self.fit_mode = None # 手動ズームに切り替え
            self.zoom_factor = min(MAX_ZOOM, self.zoom_factor * ZOOM_INCREMENT) # 最大値を超えない
            self.display_page()

    def zoom_out(self):
         """表示を縮小します。"""
         # ドキュメントが存在する場合
         if self.doc:
            self.fit_mode = None # 手動ズームに切り替え
            self.zoom_factor = max(MIN_ZOOM, self.zoom_factor / ZOOM_INCREMENT) # 最小値を下回らない
            self.display_page()

    def set_fit_width(self):
        """表示をウィンドウ幅に合わせます。"""
        # ドキュメントが存在する場合
        if self.doc:
            self.fit_mode = 'width'
            self.display_page() # ズームを再計算して表示

    def set_fit_height(self):
         """表示をウィンドウ高さに合わせます。"""
         # ドキュメントが存在する場合
         if self.doc:
             self.fit_mode = 'height'
             self.display_page() # ズームを再計算して表示

    # --- 表示モードメソッド ---
    def toggle_two_page_mode(self):
        """見開き表示モードを切り替えます。"""
        # ドキュメントが存在する場合
        if self.doc:
            self.two_page_mode = self.two_page_action.isChecked()
            # 見開き表示に切り替える際に必要であれば現在のページを調整
            # 見開きに切り替え、現在奇数ページ（1, 3, 5...）にいる場合、
            # 正しいペアを表示するために1ページ戻る（例：3にいる場合、2-3を表示）。
            # 0ページ目は display_page のロジックで正しく処理される。
            if self.two_page_mode and self.current_page > 0 and self.current_page % 2 != 0:
                 self.current_page = max(0, self.current_page - 1)

            # 表示を再計算（フィットモードで重要）
            self.display_page()

    # --- イベントハンドラ ---
    def resizeEvent(self, event):
        """ウィンドウリサイズイベントを処理します。"""
        super().resizeEvent(event)
        # リサイズ停止後に再表示を遅延させるタイマーを使用、特にフィットモードの場合
        # ドキュメントが存在し、フィットモードの場合
        if self.doc and self.fit_mode:
             current_size = self.scroll_area.viewport().size()
             # ビューポートサイズが変わった場合のみ再描画をトリガー
             if current_size != self.last_viewport_size:
                 self.last_viewport_size = current_size
                 # ドラッグ中の急速な再計算を避けるためにシングルショットタイマーを使用
                 # 必要に応じて遅延を調整
                 QTimer.singleShot(RESIZE_DELAY_MS, self.display_page)

    def keyPressEvent(self, event):
        """ショートカットのためのキープレスイベントを処理します。"""
        key = event.key()
        modifiers = event.modifiers()

        # Shift修飾子なしの場合のナビゲーション
        if modifiers == Qt.KeyboardModifier.NoModifier:
            if key == Qt.Key.Key_H:
                self.prev_page()
                event.accept()
                return
            elif key == Qt.Key.Key_L:
                self.next_page()
                event.accept()
                return
            # 他の修飾子なしキーはここで処理可能

        # ナビゲーション（QActionショートカットで処理されるが、代替を追加可能）
        # if key == Qt.Key.Key_Left:
        #     self.prev_page()
        #     event.accept() # イベントを消費
        # elif key == Qt.Key.Key_Right:
        #     self.next_page()
        #     event.accept()
        # elif key == Qt.Key.Key_Up or key == Qt.Key.Key_PageUp:
        #     # オプション: 上にスクロールまたは前のページ
        #     pass
        # elif key == Qt.Key.Key_Down or key == Qt.Key.Key_PageDown:
        #     # オプション: 下にスクロールまたは次のページ
        #     pass

        # その他のショートカット (QActionで定義済みのものは通常ここでは不要)

        # 未処理のイベントを基底クラスに渡す
        if not event.isAccepted():
             super().keyPressEvent(event)


    def closeEvent(self, event):
        """ウィンドウクローズイベントを処理します。"""
        print("クローズイベントがトリガーされました。")
        # リソースをクリーンアップ
        self._close_current_doc()
        # 実行中のAPI呼び出しがキャンセル/処理されることを確認
        if self.api_call_thread and self.api_call_thread.isRunning():
             print("終了時に実行中のAPI呼び出しをキャンセルしようとしています...")
             self._cancel_api_call() # キャンセルを試みる
             # オプションで、スレッドが終了する可能性のある非常に短い時間待機
             # self.api_call_thread.quit() # スレッド終了を要求
             # self.api_call_thread.wait(100) # 最大100ms待機

        # 終了時に設定を明示的に保存？（通常QSettingsが処理します）
        # self.settings.sync()
        print("アプリケーションを終了します。")
        event.accept() # クローズイベントを受け入れる


# --- アプリケーションエントリーポイント ---
if __name__ == "__main__":
    # 高DPIスケーリングは通常Qt6によって自動的に処理されます。
    # 属性 AA_EnableHighDpiScaling/AA_UseHighDpiPixmaps は、
    # 特定のPyQt6バージョンによっては、必要ないか利用できない場合があります。
    # スケーリングの問題が発生した場合は、OSレベルの設定または環境変数を検討してください
    # （スクリプトを実行する前に設定）例: QT_ENABLE_HIGHDPI_SCALING=1。
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True) # 必要に応じて試す
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True) # 必要に応じて試す

    app = QApplication(sys.argv)

    # オプション: アプリケーション詳細を設定（QSettingsのパスに便利）
    app.setOrganizationName("MyCompany")
    app.setApplicationName("PdfViewerGemini")

    # コマンドライン引数を確認
    initial_file = None
    if len(sys.argv) > 1:
        # 最初の引数をファイルパスとして試す
        potential_path = sys.argv[1]
        # 簡単なチェック（より堅牢なチェックも可能）
        if os.path.exists(potential_path) and potential_path.lower().endswith(".pdf"):
             initial_file = potential_path
        else:
             print(f"警告: コマンドライン引数 '{potential_path}' は有効なPDFファイルパスとして認識されませんでした。")

    # ファイルパスを渡してビューワーを初期化
    viewer = PDFViewer(initial_file_path=initial_file)
    viewer.show()
    sys.exit(app.exec())
