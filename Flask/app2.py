### buildinhgg url dynamically
###

from flask import Flask , redirect, url_for
#wsgi application - bw webserver and webapplication
app =Flask(__name__)

@app.route('/') #decorator, directs to the page(url in brackets) and this fn is executed
def welcome():
      return 'second video in the series'


@app.route('/passed/<int:score>') #decorator, directs to the page(url in brackets) and this fn is executed
def passed(score):
      return 'people passed with score' + str(score)

@app.route('/fail/<int:score>') #decorator, directs to the page(url in brackets) and this fn is executed
def fail(score):
      return 'people failed with score' + str(score)

@app.route('/results/<int:marks>')
def result(marks):
      result = ""
      if marks<50:
            result='fail'
      else:
            result= "passed"
      return redirect(url_for(result,score=marks))#redirected to passed or fail page 


if __name__ =='__main__':
    app.run(debug=True)# debug= makes it partly real time 