# server.py 是一段 Tornado 代码，让 Tornado 作为 wsgi 服务器
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from app import app
from tornado.ioloop import IOLoop
s = HTTPServer(WSGIContainer(app))
s.listen(9900)  # 监听 9900 端口
IOLoop.current().start()