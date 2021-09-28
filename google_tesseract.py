import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\zacik\AppData\Local\Tesseract-OCR\tesseract.exe'

#region Custom config for pytesseract
custom_config = r'--oem 3 --psm 6'
#endregion