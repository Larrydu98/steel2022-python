'''
hello
'''
from flask import Flask
app = Flask(__name__)

# 根目录


@app.route('/')
def hello_world():
    '''
    hello_world
    '''
    return 'index string'

# 指定url


@app.route('/hello')
def hello():
    '''
    hello
    '''
    return 'hello'

# 带参数url


@app.route('/helloname/<username>')
def show_user_name(username):
    '''
    show_user_name
    '''
    return 'hello name: %s' % username

# 带参数url，指定参数类型


@app.route('/helloid/<int:user_id>')
def show_user_id(user_id):
    '''
    show_user_id
    '''
    return 'hello id: %d' % user_id


if __name__ == '__main__':
    app.debug = True
    # 本地监听
    # app.run()
    # 监听公网ip
    app.run(host='0.0.0.0')
