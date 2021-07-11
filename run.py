'''
入口程序
'''

from alo import app

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port= 5502)
