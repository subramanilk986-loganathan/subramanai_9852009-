from src.predict import predict_heart_disease, parse_input_string
import json

def cli_loop():
    print('Heart Disease Prediction CLI. Enter a Python dict or JSON string of features. Type "quit" to exit.')
    while True:
        s = input('> ')
        if s.strip().lower() in ('quit','exit'):
            break
        try:
            data = parse_input_string(s)
            res = predict_heart_disease(data)
            print('Result:', res)
        except Exception as e:
            print('Error parsing or predicting:', e)

if __name__ == '__main__':
    cli_loop()
