import requests
import json

test_news_data = {'text':'(Reuters) - The White House said on Thursday it was focusing on getting the lowest corporate rate possible in tax reform legislation being considered on Capitol Hill. “Fifteen (percent) is better than 20, 20 is better than 22 and 22 is better than what we have,” White House spokeswoman Sarah Sanders told reporters.',
                 'title':'White House says focused on getting lowest possible corporate tax rate'}

decode_response = {'1':'True', '0':'Fake'}

print('Sending / request...')
print(f'ADDR: http://localhost:5000/predict')
r = requests.get(f'http://localhost:5000/predict', json=test_news_data)
print('Done!')
print(f'REQUEST 1: {decode_response[r.text]}')

