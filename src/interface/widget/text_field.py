from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField


class TextFieldFont(MDTextField):

  def __init__(self, **kwargs):
    self.has_had_text = False
    super().__init__(**kwargs)

  def on_font_name(self, instance, value):
    self._hint_lbl.font_name = value
    self._msg_lbl.font_name = value


class TextFieldNumeric(TextFieldFont):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.helper_text_mode = 'on_error'

  @staticmethod
  def _to_numeric(text: str):
    try:
      res = float(text)
    except ValueError:
      res = None

    return res

  def on_text(self, instance, text: str):
    super().on_text(instance, text)

    if text:
      num = self._to_numeric(text)
      if num is None:
        self.error = True
        self.helper_text = '숫자를 입력해주세요'
      else:
        self.error = False
    else:
      num = None

    return num

  def value(self):
    return self._to_numeric(self.text)


class TextFieldRatio(TextFieldNumeric):

  def on_text(self, instance, text: str):
    num = super().on_text(instance, text)

    if (num is not None) and not (0 < num <= 1):
      self.error = True
      self.helper_text = '0에서 1 사이 숫자를 입력해주세요'

    return num


class TextFieldUnit(MDBoxLayout):
  text = StringProperty('')
  hint_text = StringProperty('')
  unit = StringProperty('')

  main_text = ObjectProperty('')
  unit_text = ObjectProperty('')

  show_unit = BooleanProperty(True)
  main_text_disabled = BooleanProperty(False)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self._main_text = TextFieldNumeric()
    self._main_text.size_hint_x = 0.75

    self._unit_text = TextFieldFont()
    self._unit_text.text = self.unit
    self._unit_text.disabled = True
    self._unit_text.halign = 'right'
    self._unit_text.size_hint_x = 0.25

    self.add_widget(self._main_text)
    self.add_widget(self._unit_text)

  def get_main_text(self):
    return self._main_text.text

  def on_text(self, instance, value):
    self._main_text.text = value

  def on_hint_text(self, instance, value):
    self._main_text.hint_text = value

  def on_show_unit(self, instance, value):
    if value:
      self._unit_text.text = self.unit
    else:
      self._unit_text.text = ''

  def on_unit(self, instance, value):
    if self.show_unit:
      self._unit_text.text = value

  def on_main_text(self, instance, option: dict):
    for key, value in option.items():
      setattr(self._main_text, key, value)

  def on_unit_text(self, instance, option: dict):
    for key, value in option.items():
      setattr(self._main_text, key, value)

  def on_main_text_disabled(self, instance, value):
    self._main_text.disabled = bool(value)
