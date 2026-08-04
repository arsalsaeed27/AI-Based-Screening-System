import sys
import json
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

if __name__ == '__main__':
    image_path = sys.argv[1]
    result = reader.readtext(image_path, detail=0)
    print(json.dumps(result))
