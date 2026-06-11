""" Твърдо кодиран български токенизатор за MMS VITS """

# Директно вграждаме официалния речник (vocab) на Meta за български език
_symbol_to_id = {
  "п": 0, "е": 1, "р": 2, "–": 3, "х": 4, "щ": 5, "-": 6, "м": 7, "у": 8, "в": 9,
  "ф": 10, "ѝ": 11, "г": 12, "я": 13, "к": 14, "ц": 15, "ю": 16, "̀": 17, "и": 18, "н": 19,
  "л": 20, "з": 21, "_": 22, "й": 23, "ь": 24, "ѐ": 25, "о": 26, "с": 27, "б": 28, "ж": 29,
  "д": 30, " ": 31, "ч": 32, "ъ": 33, "т": 34, "а": 35, "ш": 36
}

# Автоматично генерираме обратното мапване (от ID към символ)
_id_to_symbol = {i: s for s, i in _symbol_to_id.items()}

def text_to_sequence(text, cleaner_names):
  ''' Превръща чистия български текст директно в ID-та, игнорирайки чистачките '''
  sequence = []
  # Превръщаме текста в малки букви, защото речникът поддържа само малки
  for symbol in text.lower():
    if symbol in _symbol_to_id:
      sequence.append(_symbol_to_id[symbol])
    else:
      # Ако срещне непознат символ, го заменя с интервал или долна черта, за да не срива процеса
      sequence.append(_symbol_to_id["_"]) 
  return sequence

def cleaned_text_to_sequence(cleaned_text):
  ''' Използва се, когато cleaned_text е изключен или подаден директно '''
  return text_to_sequence(cleaned_text, [])

def sequence_to_text(sequence):
  ''' Връща ID-тата обратно в букви (за проверка на резултата) '''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      result += _id_to_symbol[symbol_id]
  return result