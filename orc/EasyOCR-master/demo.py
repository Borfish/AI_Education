import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('demo2.jpg', detail = 0)
x = print(' '.join(result))
print(result)
