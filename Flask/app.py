from flask import Flask 
#wsgi application - bw webserver and webapplication
app =Flask(__name__)

@app.route('/') #decorator, directs to the page(url in brackets) and this fn is executed
def welcome():
      return 'hello people , ncsndjWelcome to  my youtube channel, do dubscribe'


@app.route('/members') #decorator, directs to the page(url in brackets) and this fn is executed
def members():
      return 'hello people u are in member function'



if __name__ =='__main__':
    app.run(debug=True)# debug= makes it partly real time 